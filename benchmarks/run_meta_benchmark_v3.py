#!/usr/bin/env python3
"""Meta-benchmark v3 — single-turn sweep across all single-turn envs.

Runs 3 cheap models × 2 seeds across the 6 single-turn envs that have a
working LLM adapter (sparse-Fourier, super-res, lodopab-CT, sparse-fourier-
tools skipped because it needs tool dispatch, phase-retrieval, MRI-knee).

Output: results/meta_benchmark_v3.csv (one row per episode, 3×2×6=36 rows).
Hardened: asyncio.gather(return_exceptions=True) + incremental per-episode
CSV append + hard cost cap.

Budget default: $1.30 per the sprint-giga plan.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

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
    "timestamp", "env", "model", "instance_id",
    "reward", "components", "parse_ok", "usd_cost",
    "prompt_tokens", "completion_tokens", "episode_latency_s", "error",
]

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
]

# Single-turn envs that have a working chat-completion adapter.
# Excludes sparse-fourier-tools (requires tool dispatch, benchmarked separately).
# Excludes multi-turn variants (they're implicit — meta-benchmark is single-turn only).
DEFAULT_ENVS = [
    "sparse-fourier-recovery",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "phase-retrieval",
    "mri-knee-reconstruction",
]


def _append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


async def _run_episode(client, env, adapter, env_name, model, instance_id,
                       max_tokens, timeout_s, semaphore):
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "env": env_name, "model": model, "instance_id": instance_id,
        "reward": None, "components": "", "parse_ok": False,
        "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "episode_latency_s": 0.0, "error": "",
    }
    async with semaphore:
        start = time.perf_counter()
        try:
            instance = env.generate_instance(seed=instance_id)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"generate:{type(exc).__name__}:{str(exc)[:200]}"
            return row
        messages = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    max_tokens=max_tokens, extra_body={"usage": {"include": True}},
                ),
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"api:{type(exc).__name__}:{str(exc)[:200]}"
            row["episode_latency_s"] = round(time.perf_counter() - start, 3)
            return row
        usage = completion.usage
        row["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
        row["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is not None:
            with contextlib.suppress(TypeError, ValueError):
                row["usd_cost"] = float(raw_cost)
        text = completion.choices[0].message.content or ""
        try:
            prediction = adapter.parse_response(text, instance)
            scored = env.score(prediction, instance)
        except LLMSolverError as exc:
            row["error"] = f"parse:{str(exc)[:200]}"
            row["episode_latency_s"] = round(time.perf_counter() - start, 3)
            return row
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"score:{type(exc).__name__}:{str(exc)[:200]}"
            row["episode_latency_s"] = round(time.perf_counter() - start, 3)
            return row
        row["reward"] = round(float(scored["reward"]), 4)
        row["components"] = ", ".join(
            f"{k}={float(v):.3f}" for k, v in scored["components"].items()
        )
        row["parse_ok"] = True
        row["usd_cost"] = round(float(row["usd_cost"]), 6)
        row["episode_latency_s"] = round(time.perf_counter() - start, 3)
        return row


async def _run_all(args, models, env_ids, csv_path: Path):
    if not HAS_OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/stelioszach03/verifiable-labs-envs",
            "X-Title": "verifiable-labs-envs-meta-benchmark-v3",
        },
        max_retries=3, timeout=args.timeout_s,
    )

    # Pre-load env + adapter for each env_id so we don't re-calibrate per task.
    envs = {}
    for env_id in env_ids:
        try:
            env_obj = load_environment(env_id, calibration_quantile=args.conformal_quantile)
            adapter_obj = get_adapter(env_id)
            envs[env_id] = (env_obj, adapter_obj)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load env {env_id!r}: {exc}", file=sys.stderr)

    semaphore = asyncio.Semaphore(args.max_parallel)
    tasks = []
    for env_id in envs:
        env_obj, adapter_obj = envs[env_id]
        for model in models:
            for offset in range(args.n_instances):
                tasks.append(_run_episode(
                    client, env_obj, adapter_obj, env_id, model, args.seed_start + offset,
                    args.max_tokens, args.timeout_s, semaphore,
                ))

    start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - start

    rows: list[dict] = []
    total_cost = 0.0
    for res in results:
        if isinstance(res, BaseException):
            rows.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "env": "?", "model": "?", "instance_id": -1,
                "reward": None, "components": "", "parse_ok": False,
                "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "episode_latency_s": 0.0,
                "error": f"task:{type(res).__name__}:{str(res)[:200]}",
            })
        else:
            rows.append(res)
        total_cost += float(rows[-1].get("usd_cost") or 0.0)
        _append_row(csv_path, rows[-1])
        if total_cost >= args.max_cost:
            print(f"!! running cost ${total_cost:.4f} hit cap ${args.max_cost:.2f}", file=sys.stderr)
            break
    return rows, wall, total_cost


def _print_summary(rows, models, env_ids):
    print()
    print(f"{'env':35s} | " + " | ".join(f"{m.split('/')[-1]:>28s}" for m in models))
    print("-" * (36 + len(models) * 32))
    # Aggregate per (env, model) mean
    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["parse_ok"]:
            key = (r["env"], r["model"])
            agg.setdefault(key, []).append(float(r["reward"]))
    for env_id in env_ids:
        cells = []
        for m in models:
            vals = agg.get((env_id, m), [])
            if vals:
                cells.append(f"{fmean(vals):.3f} (n={len(vals)})")
            else:
                cells.append("     —      ")
        print(f"{env_id:35s} | " + " | ".join(f"{c:>28s}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument("--n-instances", "--n", dest="n_instances", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=1.30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--conformal-quantile", type=float, default=1.587)
    parser.add_argument("--csv", type=Path,
        default=_PROJECT_ROOT / "results" / "meta_benchmark_v3.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    env_ids = [e.strip() for e in args.envs.split(",") if e.strip()]
    n_cells = len(models) * args.n_instances * len(env_ids)
    print(f"Planned: {len(env_ids)} envs × {len(models)} models × {args.n_instances} seeds = {n_cells} episodes")
    print(f"envs:     {env_ids}")
    print(f"models:   {models}")
    print(f"max_cost: ${args.max_cost:.2f}")
    print(f"csv:      {args.csv}")
    if args.dry_run:
        return 0
    rows, wall, total_cost = asyncio.run(_run_all(args, models, env_ids, args.csv))
    _print_summary(rows, models, env_ids)
    print()
    print(f"wall-clock: {wall:.1f}s  total cost: ${total_cost:.4f}  csv: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
