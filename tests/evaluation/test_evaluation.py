import json

import pytest

from trace_swe.evaluation.metrics import (
    bootstrap_mean_ci,
    cost_per_solved,
    mcnemar_exact,
    paired_bootstrap_diff,
    pass_at_k,
    wilson_interval,
)
from trace_swe.evaluation.results import RunManifest, RunResult, Score
from trace_swe.evaluation.task import TaskSet


def test_wilson_interval_contains_the_estimate():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0


def test_wilson_interval_returns_none_without_samples():
    assert wilson_interval(0, 0) is None


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson_interval(2, 1)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (1, 1, 1, 1.0),
        (10, 0, 1, 0.0),
        (10, 10, 1, 1.0),
        (5, 1, 1, 0.2),
        (2, 1, 3, None),
        (0, 0, 1, None),
    ],
)
def test_pass_at_k(n, c, k, expected):
    if expected is None:
        assert pass_at_k(n, c, k) is None
    else:
        assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    first = bootstrap_mean_ci([1, 2, 3, 4, 5], resamples=500, seed=1)
    second = bootstrap_mean_ci([1, 2, 3, 4, 5], resamples=500, seed=1)
    assert first == second
    assert first[0] <= 3.0 <= first[1]


def test_paired_bootstrap_reports_a_positive_delta():
    result = paired_bootstrap_diff([1, 1, 1, 0, 1], [0, 0, 0, 0, 0], resamples=500, seed=0)
    assert (result["delta"], result["ci_low"] > 0, result["p_value"] < 0.05) == (0.8, True, True)


def test_paired_bootstrap_requires_matching_lengths():
    assert paired_bootstrap_diff([1], [1, 0]) is None


def test_mcnemar_exact_handles_discordant_pairs():
    assert mcnemar_exact([1, 1, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.125)
    assert mcnemar_exact([1, 0], [0, 1]) == pytest.approx(1.0)
    assert mcnemar_exact([1], [1]) == pytest.approx(1.0)


def test_cost_per_solved_guards_zero_solves():
    assert cost_per_solved(0.5, 2) == 0.25
    assert cost_per_solved(0.5, 0) is None
    assert cost_per_solved(None, 2) is None


def test_task_set_loads_jsonl_and_has_a_stable_hash(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "b", "problem_statement": "fix b"}),
                json.dumps({"task_id": "a", "problem_statement": "fix a", "tags": ["smoke"]}),
            ]
        )
    )

    tasks = TaskSet.from_jsonl(path)
    assert (len(tasks), tasks.task_ids) == (2, ["b", "a"])
    assert tasks.hash == TaskSet.from_jsonl(path).hash


def test_task_set_hash_changes_with_content(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"task_id": "a", "problem_statement": "one"}))
    before = TaskSet.from_jsonl(path).hash

    path.write_text(json.dumps({"task_id": "a", "problem_statement": "two"}))
    assert TaskSet.from_jsonl(path).hash != before


def test_run_result_serializes_manifest_and_score():
    result = RunResult(
        manifest=RunManifest(
            run_id="r1",
            task_id="t1",
            task_set_hash="h1",
            model="test/model",
            started_at=1.0,
        ),
        score=Score(resolved=True, submitted=True),
        metrics={"cost": 0.1},
    )

    data = json.loads(result.to_jsonl())
    assert (data["manifest"]["task_id"], data["score"]["resolved"], data["metrics"]["cost"]) == ("t1", True, 0.1)
