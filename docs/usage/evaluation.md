# Evaluation

The evaluation layer turns trajectories into reproducible records and comparable
metrics. It has two entry points:

- `trace-extra eval-run`: run a task set end to end and write one record per attempt.
- `trace-extra eval`: aggregate saved trajectories, compare a baseline, and gate regressions.

## Task set

A task set is a JSONL file with one `TaskSpec` per line:

```json
{"task_id":"fix-value","problem_statement":"make the test pass","repo":"path/to/repo","test_command":"python -m pytest -q","timeout":60}
```

Fields:

| Field | Meaning |
|---|---|
| `task_id` | Stable identifier, also used as the output directory |
| `problem_statement` | Text passed to the agent |
| `repo` | Local repository copied into each run directory |
| `test_command` | Command that decides whether the task is resolved |
| `cwd` | Working directory relative to the copied repository |
| `protected_files` | Files the agent must not modify |
| `timeout` | Timeout for the test command |
| `tags`, `difficulty` | Optional grouping metadata |

`TaskSet` computes a stable `task_set_hash` from the sorted task content, so a
manifest can prove which task set produced a result.

## Run a task set

```bash
trace-extra eval-run tasks.jsonl \
  -o runs/my-eval \
  -m deepseek/deepseek-flash \
  -c trace.yaml \
  --agent-class verified \
  --runs 1 \
  --seed 0
```

Each attempt writes:

```text
runs/my-eval/
├── results.jsonl                 # one RunResult per line
└── <task_id>/run-01/
    ├── manifest.json             # reproducibility metadata
    ├── result.json               # manifest + score + metrics
    ├── run.traj.json             # trajectory
    └── repo/                     # isolated workspace copy
```

`manifest.json` records the task set hash, model, config/prompt hashes, package
version, git commit, seed, timestamps, exit status, task state, and artifact
paths. This is the record used to audit or replay a run.

`eval-run` currently supports local tasks. Docker tasks need a runner that scores
inside the container.

## Offline evaluation

```bash
trace-extra eval runs/my-eval \
  --baseline runs/baseline \
  --bootstrap 5000
```

The report includes:

| Metric | Meaning |
|---|---|
| `success_rate` | Strict success rate; only `task_state == "Succeeded"` counts |
| `success_ci` | Wilson 95% confidence interval |
| `cost_per_solved` | Total cost divided by resolved attempts |
| `false_completion_rate` | Submitted but unresolved, divided by submitted |
| `pass_at_k` | Per-task pass@k, using `k = min(3, runs)` |
| `paired.success_rate` | Paired bootstrap difference current - baseline |
| `paired.mcnemar_p` | Exact McNemar p-value over matched task groups |

Missing metrics stay `null` and render as `n/a`; they are never counted as zero.

## Regression gate

```bash
trace-extra eval runs/my-eval \
  --baseline runs/baseline \
  --fail-on-regression \
  --max-success-drop 0.05 \
  --max-cost-increase 0.20
```

The command exits with status 1 when the success rate drops by more than
`--max-success-drop` (5 percentage points by default) or when cost per solved
task increases by more than `--max-cost-increase` (20% by default). Use this in
CI with a small golden task set and cached model responses.

## Limitations

- `TestCommandScorer` runs the test command locally; Docker scoring is not wired yet.
- `pass@k` currently uses `k = min(3, runs)` for each task group.
- McNemar currently pairs task groups by whether the group resolved at least once;
  per-run pairing is the next refinement.
- SWE-bench results still need the official harness; `eval` reports those
  trajectories as `Unverified` until the harness report is joined.
