"""``vlabs-reward-data`` CLI entry — Phase 29.B dataset extraction.

Five subcommands wrap the
:mod:`verifiable_labs_envs.reward_distillation` package:

- ``extract`` — env-procedural rows (D4-A primary slice).
- ``extract-external`` — UltraFeedback external rows (D4-C breadth).
- ``judge`` — frontier judge slice (D5-D borderline backfill).
- ``merge`` — concatenate JSONL shards.
- ``summary`` — print aggregate stats for a JSONL file.

The CLI is intentionally thin — every command is a one-screen wrapper
over a function from the upstream package, so the test surface lives
mostly in unit tests on the upstream functions. Tests in this package
cover argument parsing + the gating behaviour (cost cap, env-var
gates, fail-fast vs continue).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from vlabs_reward_data import __version__

app = typer.Typer(
    name="vlabs-reward-data",
    help="Build the reward-distillation training dataset (Phase 29.B).",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_COST_CAP_USD: float = 30.0
"""Hard ceiling per :doc:`PHASE_29_PLAN.md` §5 D1-D. Override only via
the explicit ``--cost-cap`` flag."""


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _resolve_envs(envs_csv: str | None) -> list[str]:
    from verifiable_labs_envs.reward_distillation.dataset import default_train_envs

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
            help="Number of rows per env. Total = n_per_env * len(envs).",
        ),
    ] = 5,
    seed_start: Annotated[
        int,
        typer.Option("--seed-start", help="First seed (consecutive seeds used)."),
    ] = 0,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSONL output path."),
    ] = Path("reports/reward_distillation/extracted.jsonl"),
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--continue-on-error",
            help="Raise on first per-row error. Default is to drop the row.",
        ),
    ] = False,
) -> None:
    """Extract rows from the env catalogue (D4-A procedural slice)."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        collect_env_rows,
        dataset_summary,
        write_jsonl,
    )

    env_ids = _resolve_envs(envs)
    rows = collect_env_rows(
        env_ids=env_ids,
        n_per_env=int(n_per_env),
        seed_start=int(seed_start),
        fail_fast=bool(fail_fast),
    )
    write_jsonl(rows, output)
    summary = dataset_summary(rows)
    typer.echo(f"wrote {len(rows)} rows → {output}")
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command("extract-external")
def extract_external(
    n: Annotated[
        int,
        typer.Option("--n", min=0, help="Number of UltraFeedback rows to sample."),
    ] = 100,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 0,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSONL output path."),
    ] = Path("reports/reward_distillation/external.jsonl"),
    no_synthetic_fallback: Annotated[
        bool,
        typer.Option(
            "--no-synthetic-fallback",
            help="Hard-fail if the real dataset is unavailable.",
        ),
    ] = False,
) -> None:
    """Pull a subset of UltraFeedback (D4-C breadth slice)."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        dataset_summary,
        write_jsonl,
    )
    from verifiable_labs_envs.reward_distillation.ultrafeedback import (
        collect_ultrafeedback_subset,
    )

    rows = collect_ultrafeedback_subset(
        n=int(n),
        seed=int(seed),
        fallback_to_synthetic=not bool(no_synthetic_fallback),
    )
    write_jsonl(rows, output)
    summary = dataset_summary(rows)
    typer.echo(f"wrote {len(rows)} external rows → {output}")
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def judge(
    input: Annotated[Path, typer.Option("--input", "-i", help="Input JSONL.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output JSONL.")],
    fraction: Annotated[
        float,
        typer.Option("--fraction", min=0.0, max=1.0, help="Borderline-row fraction."),
    ] = 0.10,
    max_rows: Annotated[
        int,
        typer.Option("--max-rows", help="Hard cap on rows sent to the judge."),
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
    """Sample borderline rows and add frontier judgments."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        dataset_summary,
        is_phase29_collect_frontier_enabled,
        read_jsonl,
        write_jsonl,
    )
    from verifiable_labs_envs.reward_distillation.frontier_judge import (
        estimate_judge_cost,
        merge_judgments,
        resolve_api_key,
        sample_frontier_judgments,
        select_borderline_rows,
        stub_judge_caller,
    )

    rows = read_jsonl(input)
    selected = select_borderline_rows(
        rows, fraction=float(fraction), seed=int(seed), max_rows=int(max_rows)
    )
    estimated = estimate_judge_cost(len(selected))
    typer.echo(
        f"borderline candidates: {len(selected)} / {len(rows)} "
        f"(fraction={fraction}, max_rows={max_rows})"
    )
    typer.echo(f"estimated USD cost: ${estimated:.4f} (cap ${cost_cap_usd:.2f})")
    if estimated > cost_cap_usd:
        typer.echo(
            "ABORT: estimated cost exceeds cap; lower --fraction / --max-rows.",
            err=True,
        )
        raise typer.Exit(code=2)

    api_key = resolve_api_key()
    if force_stub or not is_phase29_collect_frontier_enabled():
        typer.echo("using stub_judge_caller (offline / gates not met)")
        results = sample_frontier_judgments(
            rows,
            fraction=float(fraction),
            judge_model=judge_model,
            api_key="<stub>",
            judge_caller=stub_judge_caller,
            seed=int(seed),
            max_rows=int(max_rows),
        )
    else:
        if not api_key:
            typer.echo("ABORT: VLABS_PHASE29_COLLECT_FRONTIER=1 but no API key.", err=True)
            raise typer.Exit(code=2)
        results = sample_frontier_judgments(
            rows,
            fraction=float(fraction),
            judge_model=judge_model,
            api_key=api_key,
            seed=int(seed),
            max_rows=int(max_rows),
        )

    merged = merge_judgments(rows, results)
    write_jsonl(merged, output)
    n_judged = sum(1 for r in merged if r.frontier_judgment is not None)
    typer.echo(
        f"wrote {len(merged)} rows ({n_judged} with frontier judgments) → {output}"
    )
    typer.echo(json.dumps(dataset_summary(merged), indent=2, sort_keys=True))


@app.command()
def merge(
    inputs: Annotated[
        str,
        typer.Option("--inputs", help="Comma-separated list of JSONL paths."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Concatenate multiple JSONL shards into a single training file."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        dataset_summary,
        merge_jsonl,
        write_jsonl,
    )

    paths = [Path(p) for p in _parse_csv(inputs)]
    if not paths:
        raise typer.BadParameter("inputs is empty")
    rows = merge_jsonl(paths)
    write_jsonl(rows, output)
    typer.echo(f"merged {len(paths)} shards into {len(rows)} rows → {output}")
    typer.echo(json.dumps(dataset_summary(rows), indent=2, sort_keys=True))


@app.command()
def summary(
    input: Annotated[Path, typer.Option("--input", "-i")],
) -> None:
    """Print the dataset summary for a JSONL file."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        dataset_summary,
        read_jsonl,
    )

    rows = read_jsonl(input)
    typer.echo(json.dumps(dataset_summary(rows), indent=2, sort_keys=True))


@app.command("extract-rewardbench")
def extract_rewardbench(
    n: Annotated[
        int,
        typer.Option("--n", min=0, help="Number of preference pairs to pull."),
    ] = 1500,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 0,
    subset: Annotated[
        str,
        typer.Option(
            "--subset",
            help="RewardBench category filter, or 'all' for the full mix.",
        ),
    ] = "all",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSONL output path."),
    ] = Path("reports/reward_distillation/rewardbench.jsonl"),
    no_synthetic_fallback: Annotated[
        bool,
        typer.Option(
            "--no-synthetic-fallback",
            help="Hard-fail if allenai/reward-bench is unreachable.",
        ),
    ] = False,
) -> None:
    """Pull preference pairs from ``allenai/reward-bench`` (D7-C cross-check).

    The output JSONL has the canonical RewardBench shape::

        {"prompt": ..., "chosen": ..., "rejected": ...,
         "category": ..., "pair_id": ..., "source": "rewardbench"}

    These rows feed the Bradley-Terry preference path in
    ``vlabs-reward-train``; they are NOT the same shape as
    :class:`RewardTrainingRow` (env-procedural rows) — the trainer
    wraps both shapes via the preference-vs-pointwise dispatch.
    """
    from verifiable_labs_envs.reward_distillation.rewardbench_adapter import (
        load_rewardbench_subset,
    )

    pairs = load_rewardbench_subset(
        n=int(n),
        seed=int(seed),
        subset=str(subset),
        fallback_to_synthetic=not bool(no_synthetic_fallback),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for p in pairs:
            row = p.to_dict()
            row["source"] = "rewardbench"
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for p in pairs:
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    typer.echo(f"wrote {len(pairs)} rewardbench pairs → {output}")
    typer.echo(json.dumps({"n_pairs": len(pairs), "per_category": by_cat}, indent=2))


@app.command()
def version() -> None:
    """Print the CLI version and exit."""
    typer.echo(f"vlabs-reward-data v{__version__}")


@app.command("env-status")
def env_status() -> None:
    """Print the resolved status of the optional gates (frontier slice
    + UltraFeedback path) so users can debug missing-key issues."""
    from verifiable_labs_envs.reward_distillation.dataset import (
        is_phase29_collect_frontier_enabled,
    )
    from verifiable_labs_envs.reward_distillation.frontier_judge import resolve_api_key

    payload = {
        "VLABS_PHASE29_COLLECT_FRONTIER": os.environ.get(
            "VLABS_PHASE29_COLLECT_FRONTIER", ""
        ),
        "OPENROUTER_API_KEY_present": bool(resolve_api_key()),
        "frontier_gate_enabled": is_phase29_collect_frontier_enabled(),
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    """Entry point used by the `vlabs-reward-data` script alias.

    Wraps :func:`app` so the binary's exit code reflects Typer's own
    exit handling cleanly under ``python -m vlabs_reward_data.cli``.
    """
    try:
        app()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
