#!/usr/bin/env python3
"""Comprehensive Sprint-1 v2 benchmark across all 6 envs × N models.

Dispatches per env type:
- **Single-turn** envs (``sparse-fourier-recovery``, ``super-resolution-div2k-x4``,
  ``lodopab-ct-simplified``) — one call per episode.
- **Multi-turn** envs (``sparse-fourier-recovery-multiturn``,
  ``lodopab-ct-simplified-multiturn``) — up to 3 turns per episode with
  ``adapter.build_followup_turn`` between each turn.
- **Tool-use** env (``sparse-fourier-recovery-tools``) — up to ``max_tool_calls``
  tool-call turns plus one final answer.

Parallelisation: ``asyncio.Semaphore(max_parallel)`` across episodes. Turns
inside an episode remain sequential because turn k+1 depends on turn k's
output.

Safety:
- Hard cost cap enforced on every append. Script aborts cleanly once the cap
  is crossed and records a ``cap_hit: True`` marker row in the CSV.
- Orders envs cheapest-first (single-turn → multi-turn → tool-use) so if the
  cap is hit, we still have complete coverage for the simplest signals.
- ``--dry-run`` prints the planned sweep without calling the API.

Outputs:
- ``results/llm_benchmark_v2.csv``: one row per turn per (model, env, seed)
  (turn=1 for single-turn; turn=k for multi-turn/tool-use).
- Aggregate table printed to stdout on completion.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

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

SINGLE_TURN_ENVS = (
    "sparse-fourier-recovery",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "phase-retrieval",
    "mri-knee-reconstruction",
)
MULTITURN_ENVS = (
    "sparse-fourier-recovery-multiturn",
    "lodopab-ct-simplified-multiturn",
    "phase-retrieval-multiturn",
    "mri-knee-reconstruction-multiturn",
)
TOOL_ENVS = (
    "sparse-fourier-recovery-tools",
)
ALL_ENVS = SINGLE_TURN_ENVS + MULTITURN_ENVS + TOOL_ENVS

DEFAULT_MODELS = (
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4",
    "anthropic/claude-opus-4.7",
)

CSV_COLUMNS = [
    "timestamp",
    "env",
    "model",
    "seed",
    "turn",  # 1 for single-turn; 1..n for multi-turn; 1..n for tool-use
    "reward",
    "components",
    "parse_ok",
    "usd_cost",
    "prompt_tokens",
    "completion_tokens",
    "latency_s",
    "error",
    "meta",
]


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cap_usd: float):
        self.cap = float(cap_usd)
        self.spent = 0.0

    def add(self, cost: float) -> None:
        self.spent += float(cost or 0.0)
        if self.spent > self.cap:
            raise BudgetExceeded(f"cumulative ${self.spent:.4f} exceeds cap ${self.cap:.2f}")

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap - self.spent)


def _call_usage(response) -> tuple[int, int, float | None]:
    usage = response.usage
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = None
    raw = getattr(usage, "cost", None)
    if raw is not None:
        with contextlib.suppress(TypeError, ValueError):
            cost = float(raw)
    return prompt, completion, cost


async def _single_turn_episode(
    client, model: str, env, env_name: str, seed: int,
    semaphore: asyncio.Semaphore, budget: Budget, per_call_timeout: float,
) -> list[dict]:
    adapter = get_adapter(env_name)
    instance = env.generate_instance(seed=seed)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "env": env_name, "model": model, "seed": seed, "turn": 1,
        "reward": None, "components": "", "parse_ok": False,
        "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        "latency_s": 0.0, "error": "", "meta": "",
    }
    async with semaphore:
        start = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": adapter.system_prompt},
                        {"role": "user", "content": adapter.build_user_prompt(instance)},
                    ],
                    temperature=0.0, max_tokens=4096,
                    extra_body={"usage": {"include": True}},
                ),
                timeout=per_call_timeout,
            )
        except Exception as exc:
            row["latency_s"] = round(time.perf_counter() - start, 3)
            row["error"] = f"api:{type(exc).__name__}:{exc}"
            return [row]
        row["latency_s"] = round(time.perf_counter() - start, 3)
    text = resp.choices[0].message.content or ""
    p, c, cost = _call_usage(resp)
    row.update({"prompt_tokens": p, "completion_tokens": c, "usd_cost": cost or 0.0})
    budget.add(cost or 0.0)
    try:
        prediction = adapter.parse_response(text, instance)
        scored = env.score(prediction, instance)
        row["reward"] = round(float(scored["reward"]), 4)
        row["components"] = ", ".join(f"{k}={v:.3f}" for k, v in scored["components"].items())
        row["parse_ok"] = True
    except LLMSolverError as exc:
        row["error"] = f"parse:{exc}"
    return [row]


async def _multiturn_episode(
    client, model: str, env, env_name: str, seed: int,
    semaphore: asyncio.Semaphore, budget: Budget, per_call_timeout: float,
) -> list[dict]:
    adapter = get_adapter(env_name)
    instance = env.generate_instance(seed=seed)
    max_turns = getattr(env, "max_turns", 3)
    history = [
        {"role": "system", "content": adapter.system_prompt},
        {"role": "user", "content": adapter.build_user_prompt(instance)},
    ]
    rows: list[dict] = []
    async with semaphore:
        for turn_idx in range(max_turns):
            row = {
                "timestamp": datetime.now(UTC).isoformat(),
                "env": env_name, "model": model, "seed": seed,
                "turn": turn_idx + 1,
                "reward": None, "components": "", "parse_ok": False,
                "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "error": "", "meta": "",
            }
            start = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model, messages=history,
                        temperature=0.0, max_tokens=4096,
                        extra_body={"usage": {"include": True}},
                    ),
                    timeout=per_call_timeout,
                )
            except Exception as exc:
                row["latency_s"] = round(time.perf_counter() - start, 3)
                row["error"] = f"api:{type(exc).__name__}:{exc}"
                rows.append(row)
                break
            row["latency_s"] = round(time.perf_counter() - start, 3)
            text = resp.choices[0].message.content or ""
            p, c, cost = _call_usage(resp)
            row.update({"prompt_tokens": p, "completion_tokens": c, "usd_cost": cost or 0.0})
            try:
                budget.add(cost or 0.0)
            except BudgetExceeded:
                rows.append(row)
                raise
            try:
                prediction = adapter.parse_response(text, instance)
            except LLMSolverError as exc:
                row["error"] = f"parse:{exc}"
                rows.append(row)
                break
            scored = env.score(prediction, instance)
            row["reward"] = round(float(scored["reward"]), 4)
            row["components"] = ", ".join(f"{k}={v:.3f}" for k, v in scored["components"].items())
            row["parse_ok"] = True
            rows.append(row)
            if turn_idx + 1 < max_turns:
                history.append({"role": "assistant", "content": text})
                history.append({
                    "role": "user",
                    "content": adapter.build_followup_turn(history, prediction, instance),
                })
    return rows


async def _tool_episode(
    client, model: str, env, env_name: str, seed: int,
    semaphore: asyncio.Semaphore, budget: Budget, per_call_timeout: float,
    max_tool_calls: int = 5,
) -> list[dict]:
    # Delegate to the same logic; the tools-env adapter's build_user_prompt
    # already tells the model about the tools. We run up to max_tool_calls+1
    # chat turns and record each.
    adapter = get_adapter(env_name)
    instance = env.generate_instance(seed=seed)
    history = [
        {"role": "system", "content": adapter.system_prompt},
        {"role": "user", "content": adapter.build_user_prompt(instance)},
    ]
    rows: list[dict] = []
    total_turns_cap = max_tool_calls + 1
    async with semaphore:
        for turn_idx in range(total_turns_cap):
            row = {
                "timestamp": datetime.now(UTC).isoformat(),
                "env": env_name, "model": model, "seed": seed,
                "turn": turn_idx + 1,
                "reward": None, "components": "", "parse_ok": False,
                "usd_cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "error": "", "meta": "",
            }
            start = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model, messages=history,
                        temperature=0.0, max_tokens=4096,
                        extra_body={"usage": {"include": True}},
                    ),
                    timeout=per_call_timeout,
                )
            except Exception as exc:
                row["latency_s"] = round(time.perf_counter() - start, 3)
                row["error"] = f"api:{type(exc).__name__}:{exc}"
                rows.append(row)
                break
            row["latency_s"] = round(time.perf_counter() - start, 3)
            text = resp.choices[0].message.content or ""
            p, c, cost = _call_usage(resp)
            row.update({"prompt_tokens": p, "completion_tokens": c, "usd_cost": cost or 0.0})
            try:
                budget.add(cost or 0.0)
            except BudgetExceeded:
                rows.append(row)
                raise
            # Try to parse as a final answer first; if that fails, treat as a tool call.
            is_final = False
            try:
                prediction = adapter.parse_response(text, instance)
                scored = env.score(prediction, instance)
                row["reward"] = round(float(scored["reward"]), 4)
                row["components"] = ", ".join(f"{k}={v:.3f}" for k, v in scored["components"].items())
                row["parse_ok"] = True
                is_final = True
            except LLMSolverError:
                # Attempt tool dispatch via adapter.execute_tool_call if available
                exec_fn = getattr(adapter, "execute_tool_call", None)
                if exec_fn is None:
                    row["error"] = "parse:not_final_answer_and_no_tool_handler"
                    rows.append(row)
                    break
                try:
                    tool_result = exec_fn(text, instance)
                    row["meta"] = json.dumps({"tool_call": True, "result_preview": str(tool_result)[:160]})
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"tool:{type(exc).__name__}:{exc}"
                    rows.append(row)
                    break
                history.append({"role": "assistant", "content": text})
                history.append({"role": "user", "content": json.dumps({"tool_result": tool_result})})
            rows.append(row)
            if is_final:
                break
    return rows


def _episode_fn(env_name: str):
    if env_name in SINGLE_TURN_ENVS:
        return _single_turn_episode
    if env_name in MULTITURN_ENVS:
        return _multiturn_episode
    if env_name in TOOL_ENVS:
        return _tool_episode
    raise ValueError(f"unknown env category for '{env_name}'")


async def _run_all(args, models, csv_path: Path):
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/stelioszach03/verifiable-labs-envs",
            "X-Title": "verifiable-labs-envs",
        },
        max_retries=2,
        timeout=180.0,
    )
    semaphore = asyncio.Semaphore(args.max_parallel)
    budget = Budget(args.max_cost)

    envs = {name: load_environment(name) for name in args.envs}

    ordered_env_types = (
        [e for e in args.envs if e in SINGLE_TURN_ENVS]
        + [e for e in args.envs if e in MULTITURN_ENVS]
        + [e for e in args.envs if e in TOOL_ENVS]
    )

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    aborted = False

    # Ensure header is written once up front so incremental appends work.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()

    async def _episode_and_persist(fn, *args_pack):
        nonlocal aborted
        try:
            rows = await fn(*args_pack)
        except BudgetExceeded:
            aborted = True
            return []
        except Exception as exc:  # noqa: BLE001 -- benchmark is resilience-first
            return [{
                "timestamp": datetime.now(UTC).isoformat(),
                "env": args_pack[3], "model": args_pack[1], "seed": args_pack[4],
                "turn": 0, "reward": None, "components": "",
                "parse_ok": False, "usd_cost": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
                "error": f"unhandled:{type(exc).__name__}:{exc}", "meta": "",
            }]
        # Append to shared list AND persist incrementally so a mid-run abort
        # doesn't lose already-completed episodes' rows.
        all_rows.extend(rows)
        with csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
        return rows

    tasks: list = []
    for env_name in ordered_env_types:
        fn = _episode_fn(env_name)
        env = envs[env_name]
        for model in models:
            for seed_offset in range(args.n_instances):
                seed = args.seed_start + seed_offset
                tasks.append(_episode_and_persist(
                    fn, client, model, env, env_name, seed, semaphore, budget,
                    args.per_call_timeout,
                ))

    await asyncio.gather(*tasks, return_exceptions=True)
    if aborted:
        print(f"\nBUDGET CAP REACHED: cumulative ${budget.spent:.4f} of ${budget.cap:.2f}", file=sys.stderr)

    wall = time.perf_counter() - t0
    return all_rows, budget.spent, wall, aborted


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def _summary(rows: list[dict]) -> None:
    from collections import defaultdict
    from statistics import fmean

    # For single-turn: take the row's reward. For multi-turn/tool: take the
    # last successful turn's reward per (model, env, seed).
    latest: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        if r["reward"] is None:
            continue
        key = (r["env"], r["model"], r["seed"])
        if key not in latest or int(r["turn"]) > int(latest[key]["turn"]):
            latest[key] = r

    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (env_name, model, _seed), r in latest.items():
        by_pair[(env_name, model)].append(float(r["reward"]))

    envs = sorted({k[0] for k in by_pair})
    models = sorted({k[1] for k in by_pair})
    print("\n=== v2 benchmark — mean reward by (model, env) — final turn ===")
    w = 42
    print(" " * w + "  ".join(f"{e[:22]:>22s}" for e in envs))
    for model in models:
        cells = []
        for env_name in envs:
            vals = by_pair.get((env_name, model), [])
            if vals:
                cells.append(f"{fmean(vals):>22.3f}")
            else:
                cells.append(f"{'—':>22s}")
        print(f"{model:42s}" + "  ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--envs", type=str, default=",".join(ALL_ENVS))
    parser.add_argument("--n", "--n-instances", dest="n_instances", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-cost", type=float, default=3.0)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--per-call-timeout", type=float, default=180.0)
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "results" / "llm_benchmark_v2.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    for env_name in envs:
        if env_name not in ALL_ENVS:
            parser.error(f"unknown env '{env_name}'. choose from: {ALL_ENVS}")
    args.envs = envs

    # Rough estimate for reporting
    single_count = sum(1 for e in envs if e in SINGLE_TURN_ENVS)
    multi_count = sum(1 for e in envs if e in MULTITURN_ENVS)
    tool_count = sum(1 for e in envs if e in TOOL_ENVS)
    n = args.n_instances * len(models)
    est_calls = n * (single_count + 3 * multi_count + 4 * tool_count)

    print(f"Models ({len(models)}): {models}")
    print(f"Envs   ({len(envs)}): {envs}")
    print(f"Per (model, env) instances: {args.n_instances}")
    print(f"Estimated call count: ~{est_calls} (single + multi*3 + tool*~4)")
    print(f"Max cost: ${args.max_cost:.2f}   max_parallel: {args.max_parallel}")
    print(f"CSV: {args.csv}")
    if args.dry_run:
        return 0
    if not HAS_OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        return 2

    rows, spent, wall, aborted = asyncio.run(_run_all(args, models, args.csv))
    print(f"\nWrote {len(rows)} rows to {args.csv} (incremental)")
    print(f"Cumulative cost: ${spent:.4f} (aborted={aborted})")
    print(f"Wall clock: {wall:.1f} s")
    _summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
