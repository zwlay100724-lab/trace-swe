"""Run a task set and persist reproducible evaluation records."""

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from trace_swe import Agent, __version__
from trace_swe.evaluation.results import RunManifest, RunResult, Score
from trace_swe.evaluation.task import TaskSet, TaskSpec


class Scorer(Protocol):
    def score(self, task: TaskSpec, trajectory: Path, workspace: Path) -> Score: ...


def prepare_workspace(task: TaskSpec, run_dir: Path) -> Path:
    """Copy a local repository into an isolated run directory."""
    if task.image:
        raise NotImplementedError(
            f"Task '{task.task_id}' uses a Docker image; the local runner does not support it yet"
        )
    if task.base_commit:
        raise NotImplementedError(f"Task '{task.task_id}' sets base_commit; the local runner does not support it yet")
    workspace = run_dir / "repo"
    if task.repo is None:
        workspace.mkdir(parents=True)
        return workspace
    source = Path(task.repo).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Task '{task.task_id}' repository not found: {source}")
    shutil.copytree(
        source,
        workspace,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv"),
    )
    return workspace


def _protected_hashes(workspace: Path, files: list[str]) -> dict[str, str | None]:
    return {
        file: hashlib.sha256((workspace / file).read_bytes()).hexdigest() if (workspace / file).is_file() else None
        for file in files
    }


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _model_name(agent: Agent) -> str:
    config = getattr(getattr(agent, "model", None), "config", None)
    return getattr(config, "model_name", type(getattr(agent, "model", agent)).__name__)


def _agent_config(agent: Agent) -> dict:
    config = getattr(agent, "config", None)
    return config.model_dump(mode="json") if hasattr(config, "model_dump") else {}


def _git_commit() -> str | None:
    if shutil.which("git") is None:
        return None
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


class TestCommandScorer:
    """Score a run by running the task's test command in the workspace."""

    __test__ = False

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout

    def score(self, task: TaskSpec, trajectory: Path, workspace: Path) -> Score:
        info = json.loads(trajectory.read_text(encoding="utf-8")).get("info", {}) if trajectory.exists() else {}
        submitted = info.get("exit_status") == "Submitted"
        if not task.test_command:
            return Score(
                resolved=info.get("task_state") == "Succeeded",
                submitted=submitted,
                failure_reason=None if info.get("task_state") == "Succeeded" else "no_test_command",
            )
        cwd = Path(task.cwd) if Path(task.cwd).is_absolute() else workspace / task.cwd
        result = subprocess.run(
            task.test_command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout or task.timeout,
        )
        resolved = result.returncode == 0
        return Score(
            resolved=resolved,
            submitted=submitted,
            failure_reason=None if resolved else "test_fail",
            metrics={"test_returncode": result.returncode},
        )


class TaskRunner:
    """Run every task/run pair and write one record per attempt."""

    def __init__(
        self,
        task_set: TaskSet,
        agent_factory: Callable[[TaskSpec, Path], Agent],
        scorer: Scorer,
        output_dir: Path,
        *,
        runs: int = 1,
        seed: int | None = None,
        prepare: Callable[[TaskSpec, Path], Path] = prepare_workspace,
    ):
        self.task_set = task_set
        self.agent_factory = agent_factory
        self.scorer = scorer
        self.output_dir = output_dir
        self.runs = runs
        self.seed = seed
        self.prepare = prepare

    def run(self) -> list[RunResult]:
        results = []
        for task in self.task_set:
            for run_index in range(1, self.runs + 1):
                results.append(self._run_one(task, run_index))
        self._write_results(results)
        return results

    def _run_one(self, task: TaskSpec, run_index: int) -> RunResult:
        run_dir = self.output_dir / task.task_id / f"run-{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        workspace = self.prepare(task, run_dir)
        protected_before = _protected_hashes(workspace, task.protected_files)
        agent = self.agent_factory(task, workspace)
        trajectory = run_dir / "run.traj.json"
        agent.config.output_path = trajectory

        started_at = time.time()
        info = agent.run(task.problem_statement)
        ended_at = time.time()
        agent.save(trajectory)

        score = self.scorer.score(task, trajectory, workspace)
        if _protected_hashes(workspace, task.protected_files) != protected_before:
            score = score.model_copy(update={"resolved": False, "failure_reason": "protected_file_modified"})
        config = _agent_config(agent)
        config_for_hash = {
            key: value
            for key, value in config.items()
            if key not in {"output_path", "event_log_path", "event_log_run_id"}
        }
        manifest = RunManifest(
            run_id=uuid.uuid4().hex,
            task_id=task.task_id,
            task_set_hash=self.task_set.hash,
            model=_model_name(agent),
            config_hash=_digest(config_for_hash),
            prompt_hash=_digest(
                {
                    "system_template": config.get("system_template"),
                    "instance_template": config.get("instance_template"),
                }
            ),
            package_version=__version__,
            git_commit=_git_commit(),
            seed=self.seed,
            started_at=started_at,
            ended_at=ended_at,
            exit_status=info.get("exit_status"),
            task_state=info.get("task_state"),
            artifacts={
                "trajectory": str(trajectory),
                "workspace": str(workspace),
            },
        )
        result = RunResult(
            manifest=manifest,
            score=score,
            metrics={
                "cost": (info.get("model_stats") or {}).get("instance_cost"),
                "n_calls": (info.get("model_stats") or {}).get("api_calls"),
            },
        )
        (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _write_results(self, results: list[RunResult]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "results.jsonl"
        path.write_text("\n".join(result.to_jsonl() for result in results) + "\n", encoding="utf-8")
