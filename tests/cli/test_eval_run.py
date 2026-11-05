import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from trace_swe.cli.commands.eval_run import app, main
from trace_swe.llm.test_models import make_output

SUBMIT = "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done"


def make_repo(path: Path) -> None:
    path.mkdir()
    (path / "buggy.py").write_text("value = 1\n")
    (path / "test_buggy.py").write_text("from buggy import value\n\ndef test_value():\n    assert value == 2\n")


def test_eval_run_help():
    result = CliRunner().invoke(app, ["--help"])
    assert (result.exit_code, "JSONL task set" in result.output) == (0, True)


def test_eval_run_executes_a_task_set(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "fix-value",
                "problem_statement": "make the test pass",
                "repo": str(repo),
                "test_command": f"{sys.executable} -m pytest -q",
                "timeout": 60,
            }
        )
    )
    outputs = [
        make_output("", [{"command": "echo 'value = 2' > buggy.py"}]),
        make_output("", [{"command": SUBMIT}]),
    ]

    main(
        tasks=tasks,
        output=tmp_path / "runs",
        model_name="deterministic",
        model_class=None,
        agent_class="default",
        config_spec=[
            "model.model_class=deterministic",
            f"model.outputs={json.dumps(outputs)}",
            "agent.system_template=system",
            "agent.instance_template={{task}}",
        ],
        runs=1,
        seed=7,
    )

    result = json.loads((tmp_path / "runs" / "results.jsonl").read_text())
    assert (result["score"]["resolved"], result["manifest"]["seed"]) == (True, 7)
    assert (tmp_path / "runs" / "fix-value" / "run-01" / "manifest.json").exists()
