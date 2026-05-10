"""``vlabs-prm-eval`` CLI entry — Phase 30.D eval harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vlabs_prm_eval import __version__

app = typer.Typer(
    name="vlabs-prm-eval",
    help="Evaluate the process reward model (Phase 30.D).",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the CLI version and exit."""
    typer.echo(f"vlabs-prm-eval v{__version__}")


@app.command("eval-processbench")
def eval_processbench(
    n: Annotated[
        int,
        typer.Option("--n", min=0, help="Number of ProcessBench traces."),
    ] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    no_synthetic_fallback: Annotated[
        bool,
        typer.Option(
            "--no-synthetic-fallback",
            help="Hard-fail if real ProcessBench is unavailable.",
        ),
    ] = False,
) -> None:
    """Run D6-A ProcessBench eval with the stub PRM."""
    from verifiable_labs_envs.process_reward.eval import (
        evaluate_processbench,
        load_processbench_subset,
    )
    from verifiable_labs_envs.process_reward.inference import stub_step_predictor

    traces = load_processbench_subset(
        n=int(n),
        seed=int(seed),
        fallback_to_synthetic=not bool(no_synthetic_fallback),
    )
    report = evaluate_processbench(traces, stub_step_predictor())
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))


@app.command("bon-rerank")
def bon_rerank(
    n: Annotated[int, typer.Option("--n", min=0)] = 10,
    n_per: Annotated[int, typer.Option("--n-per", min=1)] = 4,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """Run D6-B BoN reranking eval with the stub PRM (and stub RM
    baseline) on synthetic problems."""
    from verifiable_labs_envs.process_reward.bon_rerank import (
        make_synthetic_bon_problems,
    )
    from verifiable_labs_envs.process_reward.eval import evaluate_bon
    from verifiable_labs_envs.process_reward.inference import (
        stub_aggregate_predictor,
    )

    problems = make_synthetic_bon_problems(
        n_problems=int(n), n_per_problem=int(n_per), seed=int(seed)
    )

    def _stub_rm(prompt: str, completion: str) -> float:
        # Deterministic length-bias baseline so the BoN-vs-RM comparison
        # produces a non-trivial diff in the smoke output.
        del prompt
        return min(1.0, len(completion) / 200.0)

    metrics = evaluate_bon(
        problems,
        aggregate_predictor=stub_aggregate_predictor(),
        rm_predictor=_stub_rm,
    )
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("calibration")
def calibration(
    calib_set: Annotated[
        Path,
        typer.Option("--calib-set", help="Path to calibration JSONL."),
    ],
    target_alpha: Annotated[
        float,
        typer.Option("--target-alpha", min=0.0, max=1.0),
    ] = 0.10,
) -> None:
    """Run D9-C per-step + aggregate calibration eval."""
    from verifiable_labs_envs.process_reward.dataset import read_jsonl
    from verifiable_labs_envs.process_reward.eval import evaluate_calibration

    rows = read_jsonl(calib_set)
    if not rows:
        raise typer.BadParameter("calibration set is empty")
    result = evaluate_calibration(rows, target_alpha=float(target_alpha))
    payload = {
        "per_step_quantiles": dict(result.per_step_quantiles),
        "aggregate_quantile": float(result.aggregate_quantile),
        "aggregate_target_coverage": float(result.aggregate_target_coverage),
        "aggregate_empirical_coverage": float(
            result.aggregate_empirical_coverage
        ),
        "aggregate_drift": float(result.aggregate_drift),
        "n_traces": int(result.n_traces),
        "alpha": float(result.alpha),
        "is_calibration_suspect": result.is_calibration_suspect(),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def card(
    calib_set: Annotated[
        Path | None,
        typer.Option("--calib-set", help="Path to calibration JSONL."),
    ] = None,
    n_processbench: Annotated[
        int, typer.Option("--n-processbench", min=0)
    ] = 40,
    n_bon_problems: Annotated[
        int, typer.Option("--n-bon-problems", min=0)
    ] = 10,
    n_per_bon: Annotated[int, typer.Option("--n-per-bon", min=1)] = 4,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON output path."),
    ] = None,
) -> None:
    """Run the combined PRM eval card (ProcessBench + BoN +
    calibration) and emit the canonical JSON shape."""
    from verifiable_labs_envs.process_reward.dataset import read_jsonl
    from verifiable_labs_envs.process_reward.eval import run_eval_card

    calib_rows = read_jsonl(calib_set) if calib_set else None
    eval_card = run_eval_card(
        calib_set=calib_rows,
        n_processbench=int(n_processbench),
        n_bon_problems=int(n_bon_problems),
        n_per_bon=int(n_per_bon),
        seed=int(seed),
    )
    payload = eval_card.to_dict()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        payload["output_path"] = str(output)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["app"]
