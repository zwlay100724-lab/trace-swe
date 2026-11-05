"""Run a JSONL task set end to end and write evaluation records."""

from pathlib import Path

import typer
from rich.console import Console

from trace_swe.agent import get_agent
from trace_swe.config import get_config_from_spec
from trace_swe.evaluation import TaskRunner, TaskSet, TestCommandScorer
from trace_swe.execution import get_environment
from trace_swe.infra.serialize import UNSET, recursive_merge
from trace_swe.llm import get_model

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _agent_factory(config: dict, agent_class: str):
    def factory(task, workspace):
        model = get_model(config=config.get("model", {}))
        env = get_environment({**config.get("environment", {}), "cwd": str(workspace)}, default_type="local")
        return get_agent(model, env, config.get("agent", {}), default_type=agent_class)

    return factory


@app.command()
def main(
    tasks: Path = typer.Argument(..., help="JSONL task set"),
    output: Path = typer.Option(..., "-o", "--output", help="Directory for run records"),
    model_name: str | None = typer.Option(None, "-m", "--model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    agent_class: str = typer.Option("default", "--agent-class"),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="Config files or key=value overrides"),
    runs: int = typer.Option(1, "--runs", min=1),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    task_set = TaskSet.from_jsonl(tasks)
    if any(task.image for task in task_set):
        raise typer.BadParameter("eval-run currently supports local tasks only")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "model": {"model_name": model_name or UNSET, "model_class": model_class or UNSET},
            "agent": {"output_path": UNSET},
        }
    )
    config = recursive_merge(*configs)

    results = TaskRunner(
        task_set,
        _agent_factory(config, agent_class),
        TestCommandScorer(),
        output,
        runs=runs,
        seed=seed,
    ).run()
    resolved = sum(1 for result in results if result.score.resolved)
    console.print(f"Ran {len(results)} attempt(s): {resolved} resolved. Results: {output / 'results.jsonl'}")


if __name__ == "__main__":
    app()
