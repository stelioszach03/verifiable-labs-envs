"""``vlabs-prm-data`` CLI entry — Phase 30.B dataset extraction.

Five subcommands wrap the
:mod:`verifiable_labs_envs.process_reward` package:

- ``extract`` — env-procedural traces with per-step labels (D5-A).
- ``extend-from-rm`` — augments Phase 29 RewardTrainingRow JSONL
  with segmentation + per-step labels.
- ``judge-steps`` — frontier judge slice on borderline steps (D2-C).
- ``merge`` — concatenate JSONL shards.
- ``summary`` — print aggregate stats for a JSONL file.

The CLI is intentionally thin — every command is a one-screen wrapper
over a function from the upstream package.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from vlabs_prm_data import __version__

app = typer.Typer(
    name="vlabs-prm-data",
    help="Build the process-reward training dataset (Phase 30.B).",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_COST_CAP_USD: float = 50.0


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _resolve_envs(envs_csv: str | None) -> list[str]:
    from verifiable_labs_envs.process_reward.dataset import default_train_envs

    if envs_csv is None:
        return default_train_envs()
    parsed = _parse_csv(envs_csv)
    if not parsed:
        raise typer.BadParameter("envs list is empty")
    return parsed


@app.command()
def extract(
    envs: Annotated[
        str | None,
        typer.Option(
            "--envs",
            help="Comma-separated env ids. Defaults to the 22 training envs.",
        ),
    ] = None,
    n_per_env: Annotated[
        int,
        typer.Option(
            "--n-per-env",
            min=0,
            help="Number of traces per env. Total = n_per_env * len(envs).",
        ),
    ] = 5,
    seed_start: Annotated[
        int,
        typer.Option("--seed-start", help="First seed (consecutive seeds used)."),
    ] = 0,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Per-trace step cap."),
    ] = 32,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSONL output path."),
    ] = Path("reports/process_reward/extracted.jsonl"),
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--continue-on-error",
            help="Raise on first per-trace error. Default is to drop the trace.",
        ),
    ] = False,
) -> None:
    """Extract trace rows from the env catalogue (D5-A primary slice)."""
    from verifiable_labs_envs.process_reward.dataset import (
        collect_env_traces,
        trace_dataset_summary,
        write_jsonl,
    )

    env_ids = _resolve_envs(envs)
    rows = collect_env_traces(
        env_ids=env_ids,
        n_per_env=int(n_per_env),
        seed_start=int(seed_start),
        max_steps=int(max_steps),
        fail_fast=bool(fail_fast),
    )
    write_jsonl(rows, output)
    summary = trace_dataset_summary(rows)
    typer.echo(f"wrote {len(rows)} traces → {output}")
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("extend-from-rm")
def extend_from_rm(
    input: Annotated[
        Path,
        typer.Option(
            "--input", "-i", help="Phase 29 RewardTrainingRow JSONL path."
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 32,
) -> None:
    """Segment + relabel Phase 29 rows into PRM trace rows."""
    from verifiable_labs_envs.process_reward.dataset import (
        extend_from_phase29_rows,
        trace_dataset_summary,
        write_jsonl,
    )
    from verifiable_labs_envs.reward_distillation.dataset import read_jsonl

    rm_rows = read_jsonl(input)
    rows = extend_from_phase29_rows(rm_rows, max_steps=int(max_steps))
    write_jsonl(rows, output)
    summary = trace_dataset_summary(rows)
    typer.echo(f"extended {len(rm_rows)} RM rows → {len(rows)} PRM traces → {output}")
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("judge-steps")
def judge_steps(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    fraction: Annotated[
        float,
        typer.Option("--fraction", min=0.0, max=1.0, help="Borderline-step fraction."),
    ] = 0.10,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Hard cap on steps sent to the judge."),
    ] = 1500,
    judge_model: Annotated[
        str,
        typer.Option("--judge-model", help="OpenRouter model id."),
    ] = "anthropic/claude-sonnet-4.6",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    cost_cap_usd: Annotated[
        float,
        typer.Option("--cost-cap", help="Hard USD cap; refuses to proceed past it."),
    ] = DEFAULT_COST_CAP_USD,
    force_stub: Annotated[
        bool,
        typer.Option(
            "--force-stub",
            help="Use the deterministic stub caller even if a key is set.",
        ),
    ] = False,
) -> None:
    """Sample borderline steps and add per-step frontier judgments."""
    from verifiable_labs_envs.process_reward.dataset import (
        is_phase30_collect_frontier_enabled,
        read_jsonl,
        trace_dataset_summary,
        write_jsonl,
    )
    from verifiable_labs_envs.process_reward.frontier_judge import (
        estimate_step_judge_cost,
        merge_per_step_judgments,
        resolve_api_key,
        sample_per_step_judgments,
        select_borderline_step_targets,
        stub_step_judge_caller,
    )

    rows = read_jsonl(input)
    selected = select_borderline_step_targets(
        rows,
        fraction=float(fraction),
        seed=int(seed),
        max_steps=int(max_steps),
    )
    estimated = estimate_step_judge_cost(len(selected))
    typer.echo(
        f"borderline step targets: {len(selected)} "
        f"(fraction={fraction}, max_steps={max_steps})"
    )
    typer.echo(f"estimated USD cost: ${estimated:.4f} (cap ${cost_cap_usd:.2f})")
    if estimated > cost_cap_usd:
        typer.echo(
            "ABORT: estimated cost exceeds cap; lower --fraction / --max-steps.",
            err=True,
        )
        raise typer.Exit(code=2)

    api_key = resolve_api_key()
    if force_stub or not is_phase30_collect_frontier_enabled():
        typer.echo("using stub_step_judge_caller (offline / gates not met)")
        results = sample_per_step_judgments(
            rows,
            fraction=float(fraction),
            judge_model=judge_model,
            api_key="<stub>",
            judge_caller=stub_step_judge_caller,
            seed=int(seed),
            max_steps=int(max_steps),
        )
    else:
        if not api_key:
            typer.echo(
                "ABORT: VLABS_PHASE30_COLLECT_FRONTIER=1 but no API key.",
                err=True,
            )
            raise typer.Exit(code=2)
        results = sample_per_step_judgments(
            rows,
            fraction=float(fraction),
            judge_model=judge_model,
            api_key=api_key,
            seed=int(seed),
            max_steps=int(max_steps),
        )

    merged = merge_per_step_judgments(rows, results)
    write_jsonl(merged, output)
    judged_traces = sum(
        1
        for r in merged
        if any(j is not None for j in r.step_frontier_judgments)
    )
    typer.echo(
        f"wrote {len(merged)} traces "
        f"({judged_traces} with frontier judgments) → {output}"
    )
    typer.echo(json.dumps(trace_dataset_summary(merged), indent=2, sort_keys=True))


@app.command()
def merge(
    inputs: Annotated[str, typer.Option("--inputs")],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Concatenate multiple JSONL shards into a single training file."""
    from verifiable_labs_envs.process_reward.dataset import (
        merge_jsonl,
        trace_dataset_summary,
        write_jsonl,
    )

    paths = [Path(p) for p in _parse_csv(inputs)]
    if not paths:
        raise typer.BadParameter("inputs is empty")
    rows = merge_jsonl(paths)
    write_jsonl(rows, output)
    typer.echo(f"merged {len(paths)} shards into {len(rows)} traces → {output}")
    typer.echo(json.dumps(trace_dataset_summary(rows), indent=2, sort_keys=True))


@app.command()
def summary(
    input: Annotated[Path, typer.Option("--input", "-i")],
) -> None:
    """Print the dataset summary for a JSONL file."""
    from verifiable_labs_envs.process_reward.dataset import (
        read_jsonl,
        trace_dataset_summary,
    )

    rows = read_jsonl(input)
    typer.echo(json.dumps(trace_dataset_summary(rows), indent=2, sort_keys=True))


@app.command()
def version() -> None:
    """Print the CLI version and exit."""
    typer.echo(f"vlabs-prm-data v{__version__}")


@app.command("env-status")
def env_status() -> None:
    """Print the resolved status of the optional gates so users can
    debug missing-key issues."""
    from verifiable_labs_envs.process_reward.dataset import (
        is_phase30_collect_frontier_enabled,
    )
    from verifiable_labs_envs.process_reward.frontier_judge import resolve_api_key

    payload = {
        "VLABS_PHASE30_COLLECT_FRONTIER": os.environ.get(
            "VLABS_PHASE30_COLLECT_FRONTIER", ""
        ),
        "OPENROUTER_API_KEY_present": bool(resolve_api_key()),
        "frontier_gate_enabled": is_phase30_collect_frontier_enabled(),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["app"]
