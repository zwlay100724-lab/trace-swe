import json
import sys
from pathlib import Path

from trace_swe.agent.default import DefaultAgent
from trace_swe.evaluation.runner import TaskRunner, TestCommandScorer
from trace_swe.evaluation.task import TaskSet, TaskSpec
from trace_swe.execution.local import LocalEnvironment
from trace_swe.llm.test_models import DeterministicModel, make_output

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_repo(path: Path) -> None:
    path.mkdir()
    (path / "buggy.py").write_text("value = 1\n")
    (path / "test_buggy.py").write_text("from buggy import value\n\ndef test_value():\n    assert value == 2\n")


def make_task(repo: Path, **kwargs) -> TaskSpec:
    return TaskSpec(
        task_id="fix-value",
        problem_statement="make the test pass",
        repo=str(repo),
        test_command=f"{sys.executable} -m pytest -q",
        timeout=60,
        **kwargs,
    )


def make_agent(workspace: Path, outputs: list[dict]) -> DefaultAgent:
    return DefaultAgent(
        DeterministicModel(outputs=outputs),
        LocalEnvironment(cwd=str(workspace)),
        system_template="system",
        instance_template="{{task}}",
    )


def test_task_runner_writes_reproducible_records_and_scores(tmp_path):
    source = tmp_path / "source"
    make_repo(source)
    task_set = TaskSet([make_task(source)])

    def agent_factory(task, workspace):
        return make_agent(
            workspace,
            [
                make_output("", [{"command": "echo 'value = 2' > buggy.py"}]),
                make_output("", [{"command": SUBMIT}]),
            ],
        )

    results = TaskRunner(task_set, agent_factory, TestCommandScorer(), tmp_path / "runs").run()

    assert (len(results), results[0].score.resolved, results[0].score.submitted) == (1, True, True)
    assert results[0].manifest.task_set_hash == task_set.hash
    run_dir = tmp_path / "runs" / "fix-value" / "run-01"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "run.traj.json").exists()
    assert (tmp_path / "runs" / "results.jsonl").read_text().count("\n") == 1
    assert (run_dir / "repo" / "buggy.py").read_text().strip() == "value = 2"
    assert (source / "buggy.py").read_text().strip() == "value = 1"


def test_task_runner_records_a_failed_score(tmp_path):
    source = tmp_path / "source"
    make_repo(source)

    def agent_factory(task, workspace):
        return make_agent(workspace, [make_output("", [{"command": SUBMIT}])])

    results = TaskRunner(TaskSet([make_task(source)]), agent_factory, TestCommandScorer(), tmp_path / "runs").run()

    assert (results[0].score.resolved, results[0].score.failure_reason) == (False, "test_fail")
    assert json.loads((tmp_path / "runs" / "results.jsonl").read_text())["score"]["resolved"] is False


def test_task_runner_rejects_protected_file_changes(tmp_path):
    source = tmp_path / "source"
    make_repo(source)
    (source / "passing_test.py").write_text("from buggy import value\n\ndef test_value():\n    assert value == 1\n")

    def agent_factory(task, workspace):
        return make_agent(
            workspace,
            [
                make_output("", [{"command": "cp passing_test.py test_buggy.py"}]),
                make_output("", [{"command": SUBMIT}]),
            ],
        )

    task = make_task(source, protected_files=["test_buggy.py"])
    results = TaskRunner(TaskSet([task]), agent_factory, TestCommandScorer(), tmp_path / "runs").run()

    assert (results[0].score.resolved, results[0].score.failure_reason) == (False, "protected_file_modified")
