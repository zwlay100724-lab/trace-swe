"""Stable records for one evaluation run."""

from typing import Any

from pydantic import BaseModel


class Score(BaseModel):
    resolved: bool = False
    submitted: bool = False
    patch_applied: bool | None = None
    tests_passed: int | None = None
    tests_failed: int | None = None
    regressions: int | None = None
    failure_reason: str | None = None
    metrics: dict[str, float | int | None] = {}


class RunManifest(BaseModel):
    """Everything needed to reproduce and audit a run."""

    run_id: str
    task_id: str
    task_set_hash: str
    model: str
    model_params: dict[str, Any] = {}
    prompt_hash: str | None = None
    config_hash: str | None = None
    package_version: str = ""
    git_commit: str | None = None
    dataset_version: str | None = None
    image_digest: str | None = None
    seed: int | None = None
    started_at: float
    ended_at: float | None = None
    exit_status: str | None = None
    task_state: str | None = None
    artifacts: dict[str, str] = {}


class RunResult(BaseModel):
    manifest: RunManifest
    score: Score
    metrics: dict[str, float | int | None] = {}

    def to_jsonl(self) -> str:
        return self.model_dump_json()
