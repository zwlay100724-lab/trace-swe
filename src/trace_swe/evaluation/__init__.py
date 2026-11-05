"""Evaluation primitives: tasks, run manifests, scores, and metrics."""

from trace_swe.evaluation.metrics import (
    bootstrap_mean_ci,
    cost_per_solved,
    mcnemar_exact,
    paired_bootstrap_diff,
    pass_at_k,
    wilson_interval,
)
from trace_swe.evaluation.results import RunManifest, RunResult, Score
from trace_swe.evaluation.runner import Scorer, TaskRunner, TestCommandScorer, prepare_workspace
from trace_swe.evaluation.task import TaskSet, TaskSpec

__all__ = [
    "RunManifest",
    "RunResult",
    "Score",
    "Scorer",
    "TaskRunner",
    "TaskSet",
    "TaskSpec",
    "TestCommandScorer",
    "prepare_workspace",
    "bootstrap_mean_ci",
    "cost_per_solved",
    "mcnemar_exact",
    "paired_bootstrap_diff",
    "pass_at_k",
    "wilson_interval",
]
