#!/usr/bin/env python3
"""Async benchmark for the MRI-knee-reconstruction env (sprint-giga Task 2).

3 models × 3 seeds × 2 variants (single, multi-turn), $0.60 cap, incremental
CSV, hardened asyncio.gather(return_exceptions=True).
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
    "timestamp", "env", "env_version", "variant", "model", "instance_id",
    "reward", "components", "n_turns", "parse_ok", "usd_cost",
    "prompt_tokens", "completion_tokens", "episode_latency_s", "error",
]

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
]


def _append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


async def _run_single(client, env, adapter, model, instance_id, max_tokens, timeout_s):
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "env": env.name, "env_version": "1.0.0", "variant": "single",
        "model": model, "instance_id": instance_id,
        "reward": None, "components": "", "n_turns": 1, "parse_ok": False,
        "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "episode_latency_s": 0.0, "error": "",
    }
    start = time.perf_counter()
    instance = env.generate_instance(seed=instance_id)
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
    row["reward"] = round(float(scored["reward"]), 4)
    row["components"] = ", ".join(f"{k}={float(v):.3f}" for k, v in scored["components"].items())
    row["parse_ok"] = True
    row["usd_cost"] = round(float(row["usd_cost"]), 6)
    row["episode_latency_s"] = round(time.perf_counter() - start, 3)
    return row


async def _run_mt(client, env, adapter, model, instance_id, max_tokens, timeout_s):
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "env": env.name, "env_version": "1.0.0", "variant": "multiturn",
        "model": model, "instance_id": instance_id,
        "reward": None, "components": "", "n_turns": 0, "parse_ok": False,
        "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "episode_latency_s": 0.0, "error": "",
    }
    start = time.perf_counter()
    instance = env.generate_instance(seed=instance_id)
    messages = [
        {"role": "system", "content": adapter.system_prompt},
        {"role": "user", "content": adapter.build_user_prompt(instance)},
    ]
    last_pred = None
    for turn_idx in range(3):
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model, messages=messages, temperature=0.0,
                    max_tokens=max_tokens, extra_body={"usage": {"include": True}},
                ),
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"api:turn{turn_idx}:{type(exc).__name__}:{str(exc)[:150]}"
            break
        usage = completion.usage
        row["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        row["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is not None:
            with contextlib.suppress(TypeError, ValueError):
                row["usd_cost"] = float(row["usd_cost"]) + float(raw_cost)
        text = completion.choices[0].message.content or ""
        try:
            prediction = adapter.parse_response(text, instance)
        except LLMSolverError as exc:
            if last_pred is None:
                row["error"] = f"parse:turn{turn_idx}:{str(exc)[:200]}"
            break
        last_pred = prediction
        row["n_turns"] = turn_idx + 1
        if turn_idx < 2:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": adapter.build_followup_turn(messages, prediction, instance)})
    if last_pred is not None:
        scored = env.score(last_pred, instance)
        row["reward"] = round(float(scored["reward"]), 4)
        row["components"] = ", ".join(f"{k}={float(v):.3f}" for k, v in scored["components"].items())
        row["parse_ok"] = True
    row["usd_cost"] = round(float(row["usd_cost"]), 6)
    row["episode_latency_s"] = round(time.perf_counter() - start, 3)
    return row


async def _run_all(args, models, csv_path: Path):
    if not HAS_OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/stelioszach03/verifiable-labs-envs",
            "X-Title": "verifiable-labs-envs-mri-knee-v1",
        },
        max_retries=3, timeout=args.timeout_s,
    )
    env_single = load_environment("mri-knee-reconstruction", calibration_quantile=args.conformal_quantile)
    adapter_single = get_adapter("mri-knee-reconstruction")
    env_mt = load_environment("mri-knee-reconstruction-multiturn", calibration_quantile=args.conformal_quantile)
    adapter_mt = get_adapter("mri-knee-reconstruction-multiturn")
    semaphore = asyncio.Semaphore(args.max_parallel)
    async def _guarded(coro):
        async with semaphore:
            return await coro
    tasks = []
    for model in models:
        for offset in range(args.n_instances):
            if "single" in args.variants:
                tasks.append(_guarded(_run_single(client, env_single, adapter_single, model, args.seed_start + offset, args.max_tokens, args.timeout_s)))
            if "multiturn" in args.variants:
                tasks.append(_guarded(_run_mt(client, env_mt, adapter_mt, model, args.seed_start + offset, args.max_tokens, args.timeout_s)))
    start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - start
    rows: list[dict] = []
    total_cost = 0.0
    for res in results:
        if isinstance(res, BaseException):
            rows.append({
                "timestamp": datetime.now(UTC).isoformat(), "env": "mri-knee-reconstruction",
                "env_version": "1.0.0", "variant": "?", "model": "?", "instance_id": -1,
                "reward": None, "components": "", "n_turns": 0, "parse_ok": False,
                "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "episode_latency_s": 0.0, "error": f"task:{type(res).__name__}:{str(res)[:200]}",
            })
        else:
            rows.append(res)
        total_cost += float(rows[-1].get("usd_cost") or 0.0)
        _append_row(csv_path, rows[-1])
        if total_cost >= args.max_cost:
            print(f"!! running cost ${total_cost:.4f} hit cap ${args.max_cost:.2f}", file=sys.stderr)
            break
    return rows, wall, total_cost


def _print_summary(rows, models):
    print()
    print(f"{'model':40s}  {'variant':>10s}  {'inst':>4s}  {'rew':>6s}  {'turns':>5s}  {'cost':>8s}  {'err':>4s}")
    print("-" * 105)
    for r in rows:
        reward = f"{r['reward']:.3f}" if r["parse_ok"] else "FAIL"
        turns = r.get("n_turns", "-")
        cost = f"${float(r['usd_cost'] or 0):.4f}"
        err = "Y" if not r["parse_ok"] else ""
        print(f"{r['model']:40s}  {r['variant']:>10s}  {r['instance_id']!s:>4s}  {reward:>6s}  {turns!s:>5s}  {cost:>8s}  {err:>4s}")
    print()
    for model in models:
        for variant in ("single", "multiturn"):
            vals = [float(r["reward"]) for r in rows if r["model"] == model and r["variant"] == variant and r["parse_ok"]]
            if vals:
                print(f"  {model:40s} {variant:>10s}  n={len(vals)}  mean={fmean(vals):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--variants", type=str, default="single,multiturn")
    parser.add_argument("--n-instances", "--n", dest="n_instances", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--max-cost", type=float, default=0.60)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--conformal-quantile", type=float, default=2.0)
    parser.add_argument("--csv", type=Path,
        default=_PROJECT_ROOT / "results" / "mri_knee_v1_benchmark.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n_cells = len(models) * args.n_instances * len(args.variants)
    print(f"Planned: {len(models)} models × {args.n_instances} seeds × {len(args.variants)} variants = {n_cells} episodes")
    print(f"max_cost: ${args.max_cost:.2f}")
    print(f"csv: {args.csv}")
    if args.dry_run:
        return 0
    rows, wall, total_cost = asyncio.run(_run_all(args, models, args.csv))
    _print_summary(rows, models)
    print()
    print(f"wall-clock: {wall:.1f}s  total cost: ${total_cost:.4f}  csv: {args.csv}")
    if total_cost > args.max_cost:
        print(f"!! exceeded cap ${args.max_cost:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
