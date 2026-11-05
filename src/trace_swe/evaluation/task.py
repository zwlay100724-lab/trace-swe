"""Task definitions shared by local experiments and benchmark adapters."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


class TaskSpec(BaseModel):
    """One reproducible evaluation task."""

    task_id: str
    problem_statement: str
    test_command: str = ""
    repo: str | None = None
    base_commit: str | None = None
    image: str | None = None
    cwd: str = ""
    protected_files: list[str] = []
    timeout: int = 300
    tags: list[str] = []
    difficulty: str | None = None


class TaskSet:
    """An ordered collection of tasks with a stable content hash."""

    def __init__(self, tasks: list[TaskSpec]):
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, path: Path) -> "TaskSet":
        tasks = [
            TaskSpec.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(tasks)

    @property
    def task_ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]

    @property
    def hash(self) -> str:
        payload = [task.model_dump(mode="json") for task in sorted(self.tasks, key=lambda task: task.task_id)]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)
