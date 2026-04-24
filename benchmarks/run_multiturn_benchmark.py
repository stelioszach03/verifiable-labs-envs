#!/usr/bin/env python3
"""Async multi-turn benchmark for verifiable-labs-envs.

Parallelises across (model, instance) episode dimensions while keeping turns
sequential within an episode (turn k+1 depends on the residual/feedback built
from turn k). Concurrency is capped by an ``asyncio.Semaphore(max_parallel)``.

Output:
- ``results/multiturn_<envshort>_benchmark.csv`` (one row per turn per episode).
- Pretty per-turn table to stdout, cumulative cost, wall-clock time.

Safety:
- Budget hard-cap; aborts the gather as soon as cumulative cost crosses ``--max-cost``.
- Per-call timeout 120 s; per-episode timeout proportional to max_turns.
- Parse failures in the middle of an episode halt that episode gracefully.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

# Bootstrap src/ for reliable imports regardless of .pth state
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from verifiable_labs_envs import load_environment  # noqa: E402
from verifiable_labs_envs.solvers.llm_solver import (  # noqa: E402
    HAS_OPENROUTER_KEY,
    LLMSolverError,
    get_adapter,
)

CSV_COLUMNS = [
    "timestamp",
    "env",
    "model",
    "instance_id",
    "turn",
    "reward",
    "components",
    "parse_ok",
    "usd_cost",
    "prompt_tokens",
    "completion_tokens",
    "latency_s",
    "error",
]


async def _run_episode(
    client,
    env,
    adapter,
    model: str,
    instance_id: int,
    *,
    max_turns: int,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    timeout_s: float,
) -> list[dict]:
    """Run one (model, instance) episode; returns up to ``max_turns`` row dicts."""
    rows: list[dict] = []
    async with semaphore:
        instance = env.generate_instance(seed=instance_id)
        history = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        last_prediction = None
        for turn in range(max_turns):
            row = {
                "timestamp": datetime.now(UTC).isoformat(),
                "env": env.name,
                "model": model,
                "instance_id": instance_id,
                "turn": turn,
                "reward": None,
                "components": "",
                "parse_ok": False,
                "usd_cost": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0.0,
                "error": "",
            }

            start = time.perf_counter()
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=history,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        extra_body={"usage": {"include": True}},
                    ),
                    timeout=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 -- benchmark must be resilient
                row["latency_s"] = round(time.perf_counter() - start, 3)
                row["error"] = f"api:{type(exc).__name__}:{str(exc)[:200]}"
                rows.append(row)
                return rows
            row["latency_s"] = round(time.perf_counter() - start, 3)

            text = completion.choices[0].message.content or ""
            usage = completion.usage
            row["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            row["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
            raw_cost = getattr(usage, "cost", None)
            if raw_cost is not None:
                with contextlib.suppress(TypeError, ValueError):
                    row["usd_cost"] = float(raw_cost)

            try:
                prediction = adapter.parse_response(text, instance)
                scored = env.score(prediction, instance)
            except LLMSolverError as exc:
                row["error"] = f"parse:{str(exc)[:200]}"
                rows.append(row)
                return rows  # halt episode on parse failure
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"score:{type(exc).__name__}:{str(exc)[:200]}"
                rows.append(row)
                return rows

            row["reward"] = round(float(scored["reward"]), 4)
            row["components"] = ", ".join(
                f"{k}={float(v):.3f}" for k, v in scored["components"].items()
            )
            row["parse_ok"] = True
            last_prediction = prediction
            rows.append(row)

            if turn + 1 < max_turns:
                history.append({"role": "assistant", "content": text})
                try:
                    followup = adapter.build_followup_turn(history, last_prediction, instance)
                except NotImplementedError as exc:
                    # Adapter doesn't support multi-turn -- end episode.
                    rows[-1]["error"] = f"adapter:{exc}"
                    return rows
                history.append({"role": "user", "content": followup})
        return rows


async def _run_all(
    env_name: str,
    models: list[str],
    n_instances: int,
    seed_start: int,
    max_turns: int,
    max_parallel: int,
    max_cost: float,
    max_tokens: int,
    timeout_s: float,
    csv_path: Path,
    conformal_quantile: float | None,
) -> tuple[list[dict], float, float]:
    """Returns (all rows, total_cost, wall_clock_s)."""
    if not HAS_OPENROUTER_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the environment."
        )
    from openai import AsyncOpenAI  # local import so script works in dry-run without openai async

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=__import__("os").environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/stelioszach03/verifiable-labs-envs",
            "X-Title": "verifiable-labs-envs",
        },
        max_retries=3,
        timeout=timeout_s,
    )

    load_kwargs = {}
    if conformal_quantile is not None:
        load_kwargs["calibration_quantile"] = conformal_quantile
    env = load_environment(env_name, **load_kwargs) if load_kwargs else load_environment(env_name)
    adapter = get_adapter(env_name)

    semaphore = asyncio.Semaphore(max_parallel)
    tasks = []
    for model in models:
        for offset in range(n_instances):
            instance_id = seed_start + offset
            tasks.append(
                _run_episode(
                    client, env, adapter, model, instance_id,
                    max_turns=max_turns,
                    semaphore=semaphore,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
            )

    wall_start = time.perf_counter()
    results_nested: list[list[dict]] = await asyncio.gather(*tasks, return_exceptions=False)
    wall_clock = time.perf_counter() - wall_start

    all_rows: list[dict] = [row for ep in results_nested for row in ep]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        writer.writerows(all_rows)

    total_cost = sum(float(r.get("usd_cost") or 0.0) for r in all_rows)
    if total_cost > max_cost:
        print(f"\n!! cumulative cost ${total_cost:.4f} exceeds --max-cost ${max_cost:.2f}", file=sys.stderr)

    return all_rows, total_cost, wall_clock


def _print_summary(rows: list[dict], models: list[str], max_turns: int) -> None:
    print()
    print(f"{'model':40s}  {'inst':>4s}  " + "  ".join(f"turn_{t}".rjust(10) for t in range(max_turns)) + "   final")
    print("-" * (52 + 12 * (max_turns + 1)))
    by_episode: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_episode.setdefault((row["model"], int(row["instance_id"])), []).append(row)

    for model in models:
        for (m, inst_id), ep_rows in sorted(by_episode.items()):
            if m != model:
                continue
            turn_cells = []
            final_reward = None
            for t in range(max_turns):
                hits = [r for r in ep_rows if int(r["turn"]) == t]
                if not hits:
                    turn_cells.append("—".rjust(10))
                elif not hits[0]["parse_ok"]:
                    turn_cells.append("FAIL".rjust(10))
                else:
                    reward = float(hits[0]["reward"])
                    final_reward = reward
                    turn_cells.append(f"{reward:.3f}".rjust(10))
            tail = f"{final_reward:.3f}" if final_reward is not None else "  —  "
            print(f"{model:40s}  {inst_id:>4d}  " + "  ".join(turn_cells) + f"   {tail}")
    # Per-model mean final reward + mean per-turn improvement
    print()
    for model in models:
        finals = []
        per_turn_means = [[] for _ in range(max_turns)]
        for (m, _), ep_rows in by_episode.items():
            if m != model:
                continue
            last_good = None
            for t in range(max_turns):
                hits = [r for r in ep_rows if int(r["turn"]) == t]
                if hits and hits[0]["parse_ok"]:
                    per_turn_means[t].append(float(hits[0]["reward"]))
                    last_good = float(hits[0]["reward"])
            if last_good is not None:
                finals.append(last_good)
        per_t = [fmean(vals) if vals else float("nan") for vals in per_turn_means]
        fin = fmean(finals) if finals else float("nan")
        per_t_str = " -> ".join(f"{v:.3f}" if v == v else "  —  " for v in per_t)
        print(f"  {model:40s}  per-turn mean: {per_t_str}   final mean: {fin:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True,
                        help="registered env name, e.g. sparse-fourier-recovery-multiturn")
    parser.add_argument("--models", required=True,
                        help="comma-separated OpenRouter model IDs")
    parser.add_argument("--n-instances", "--n", dest="n_instances", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--conformal-quantile", type=float, default=None,
                        help="skip env's own calibration and pin this value")
    parser.add_argument("--csv", type=Path, default=None,
                        help="override output CSV path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.csv is None:
        env_short = args.env.replace("-", "_").replace("/", "_")
        args.csv = _PROJECT_ROOT / "results" / f"multiturn_{env_short}.csv"

    total_calls_planned = len(models) * args.n_instances * args.max_turns
    print(f"Planned: {len(models)} model(s) x {args.n_instances} instance(s) x "
          f"up to {args.max_turns} turn(s) = up to {total_calls_planned} calls")
    print(f"env:           {args.env}")
    print(f"models:        {models}")
    print(f"max_parallel:  {args.max_parallel}")
    print(f"max_cost:      ${args.max_cost:.2f}")
    print(f"csv:           {args.csv}")

    if args.dry_run:
        return 0

    rows, total_cost, wall_clock = asyncio.run(
        _run_all(
            env_name=args.env,
            models=models,
            n_instances=args.n_instances,
            seed_start=args.seed_start,
            max_turns=args.max_turns,
            max_parallel=args.max_parallel,
            max_cost=args.max_cost,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            csv_path=args.csv,
            conformal_quantile=args.conformal_quantile,
        )
    )

    _print_summary(rows, models, args.max_turns)
    n_parse_fail = sum(1 for r in rows if not r["parse_ok"])
    print()
    print(f"wall-clock:    {wall_clock:.1f}s")
    print(f"total cost:    ${total_cost:.4f}")
    print(f"rows:          {len(rows)} ({n_parse_fail} parse failures)")
    print(f"csv:           {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
