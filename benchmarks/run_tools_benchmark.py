#!/usr/bin/env python3
"""Async tool-use benchmark for sparse-fourier-recovery-tools.

Parallelises at the (model, instance) episode level via asyncio.Semaphore.
Each episode drives up to ``max_tool_calls`` tool-call loops and then a final
no-tool answer. Tools are dispatched server-side via the env's
``dispatch_tool`` helper; tool results are appended as ``tool``-role messages
before the next LLM turn.

Output:
- CSV with one row per episode (model, instance, n_tool_calls, tool_sequence,
  final_reward, components, parse_ok, total_episode_cost, episode_latency).
- Stdout summary table.
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
from statistics import fmean

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from verifiable_labs_envs import load_environment  # noqa: E402
from verifiable_labs_envs.envs.sparse_fourier_tools import (  # noqa: E402
    TOOL_SCHEMAS,
    dispatch_tool,
)
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
    "reward",
    "components",
    "tool_calls",
    "tool_sequence",
    "parse_ok",
    "usd_cost",
    "prompt_tokens",
    "completion_tokens",
    "episode_latency_s",
    "n_llm_calls",
    "error",
]


def _normalize_tool_calls(raw) -> list[dict]:
    calls = []
    for tc in raw or []:
        calls.append({
            "id": getattr(tc, "id", "") or tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
            "type": "function",
            "function": {
                "name": (tc.function.name if hasattr(tc, "function") else tc["function"]["name"]),
                "arguments": (tc.function.arguments if hasattr(tc, "function") else tc["function"]["arguments"]),
            },
        })
    return calls


async def _run_episode(
    client,
    env,
    adapter,
    model: str,
    instance_id: int,
    *,
    max_tool_calls: int,
    max_loops: int,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "env": env.name,
        "model": model,
        "instance_id": instance_id,
        "reward": None,
        "components": "",
        "tool_calls": 0,
        "tool_sequence": "",
        "parse_ok": False,
        "usd_cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "episode_latency_s": 0.0,
        "n_llm_calls": 0,
        "error": "",
    }

    async with semaphore:
        episode_start = time.perf_counter()
        instance = env.generate_instance(seed=instance_id)
        messages: list[dict] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls_total = 0
        tool_sequence: list[str] = []
        final_text: str | None = None

        for _loop in range(max_loops):
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        tools=TOOL_SCHEMAS,
                        extra_body={"usage": {"include": True}},
                    ),
                    timeout=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"api:{type(exc).__name__}:{str(exc)[:200]}"
                row["episode_latency_s"] = round(time.perf_counter() - episode_start, 3)
                return row
            row["n_llm_calls"] += 1

            usage = completion.usage
            row["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            row["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            raw_cost = getattr(usage, "cost", None)
            if raw_cost is not None:
                with contextlib.suppress(TypeError, ValueError):
                    row["usd_cost"] = float(row["usd_cost"]) + float(raw_cost)

            message = completion.choices[0].message
            text = message.content or ""
            tool_calls_sdk = getattr(message, "tool_calls", None)

            if tool_calls_sdk and tool_calls_total < max_tool_calls:
                remaining = max_tool_calls - tool_calls_total
                tool_calls_sdk_limited = tool_calls_sdk[:remaining]
                asst_calls_payload = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in tool_calls_sdk_limited]
                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": asst_calls_payload,
                })
                for tc in tool_calls_sdk_limited:
                    name = tc.function.name
                    args_str = tc.function.arguments
                    tool_result = dispatch_tool(name, args_str, instance)
                    tool_calls_total += 1
                    tool_sequence.append(name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result),
                    })
                continue

            if tool_calls_sdk and tool_calls_total >= max_tool_calls:
                asst_calls_payload = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in tool_calls_sdk]
                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": asst_calls_payload,
                })
                for tc in tool_calls_sdk:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "tool-call budget exhausted; output final JSON answer now"}),
                    })
                messages.append({
                    "role": "user",
                    "content": "Tool-call budget exhausted. Output your final JSON answer now (support_idx, support_amp_x1000). No tool calls.",
                })
                continue

            # No tool calls -> final answer
            final_text = text
            break

        row["tool_calls"] = tool_calls_total
        row["tool_sequence"] = ",".join(tool_sequence)

        if final_text is None:
            row["error"] = f"no-final-answer-after-{max_loops}-loops"
            row["episode_latency_s"] = round(time.perf_counter() - episode_start, 3)
            return row

        try:
            prediction = adapter.parse_response(final_text, instance)
            scored = env.score(prediction, instance)
        except LLMSolverError as exc:
            row["error"] = f"parse:{str(exc)[:200]}"
            row["episode_latency_s"] = round(time.perf_counter() - episode_start, 3)
            return row

        row["reward"] = round(float(scored["reward"]), 4)
        row["components"] = ", ".join(
            f"{k}={float(v):.3f}" for k, v in scored["components"].items()
        )
        row["parse_ok"] = True
        row["usd_cost"] = round(float(row["usd_cost"]), 6)
        row["episode_latency_s"] = round(time.perf_counter() - episode_start, 3)
        return row


async def _run_all(args, models):
    if not HAS_OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=__import__("os").environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https://github.com/stelioszach03/verifiable-labs-envs",
            "X-Title": "verifiable-labs-envs",
        },
        max_retries=3,
        timeout=args.timeout_s,
    )
    env = load_environment(args.env, calibration_quantile=args.conformal_quantile)
    adapter = get_adapter(args.env)
    semaphore = asyncio.Semaphore(args.max_parallel)

    tasks = []
    for model in models:
        for offset in range(args.n_instances):
            tasks.append(_run_episode(
                client, env, adapter, model, args.seed_start + offset,
                max_tool_calls=args.max_tool_calls,
                max_loops=args.max_loops,
                semaphore=semaphore,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
            ))

    start = time.perf_counter()
    rows = await asyncio.gather(*tasks)
    wall = time.perf_counter() - start

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not args.csv.exists()
    with args.csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        writer.writerows(rows)

    return rows, wall


def _print_summary(rows, models):
    print()
    print(f"{'model':40s}  {'inst':>4s}  {'tools':>5s}  {'reward':>8s}  {'cost':>8s}  {'latency':>8s}  seq")
    print("-" * 100)
    for row in rows:
        model = row["model"]
        inst = int(row["instance_id"])
        tools = int(row["tool_calls"])
        reward = f"{row['reward']:.3f}" if row["parse_ok"] else "FAIL"
        cost = f"${float(row['usd_cost']):.4f}"
        latency = f"{float(row['episode_latency_s']):.1f}s"
        seq = row["tool_sequence"] or "(no tools)"
        print(f"{model:40s}  {inst:>4d}  {tools:>5d}  {reward:>8s}  {cost:>8s}  {latency:>8s}  {seq}")
    print()
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        successful = [float(r["reward"]) for r in model_rows if r["parse_ok"]]
        mean_reward = fmean(successful) if successful else float("nan")
        total_cost = sum(float(r["usd_cost"] or 0) for r in model_rows)
        total_tools = sum(int(r["tool_calls"] or 0) for r in model_rows)
        fails = sum(1 for r in model_rows if not r["parse_ok"])
        mean_str = f"{mean_reward:.3f}" if mean_reward == mean_reward else "  —  "
        print(
            f"  {model:40s}  final mean: {mean_str}  "
            f"total tools: {total_tools}  total cost: ${total_cost:.4f}  fails: {fails}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="sparse-fourier-recovery-tools")
    parser.add_argument("--models", required=True)
    parser.add_argument("--n-instances", "--n", dest="n_instances", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--max-loops", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=10)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--conformal-quantile", type=float, default=1.587)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.csv is None:
        env_short = args.env.replace("-", "_").replace("/", "_")
        args.csv = _PROJECT_ROOT / "results" / f"tools_{env_short}.csv"

    max_calls_per_episode = args.max_tool_calls + 1  # +1 for final answer
    total_calls_max = len(models) * args.n_instances * max_calls_per_episode
    print(f"Planned: {len(models)} model(s) x {args.n_instances} instance(s) x up to "
          f"{max_calls_per_episode} LLM calls = up to {total_calls_max} calls")
    print(f"env:           {args.env}")
    print(f"models:        {models}")
    print(f"max_tool_calls per episode: {args.max_tool_calls}")
    print(f"max_parallel:  {args.max_parallel}")
    print(f"max_cost:      ${args.max_cost:.2f}")
    print(f"csv:           {args.csv}")

    if args.dry_run:
        return 0

    rows, wall = asyncio.run(_run_all(args, models))
    _print_summary(rows, models)
    total_cost = sum(float(r["usd_cost"] or 0) for r in rows)
    print()
    print(f"wall-clock:    {wall:.1f}s")
    print(f"total cost:    ${total_cost:.4f}")
    print(f"csv:           {args.csv}")
    if total_cost > args.max_cost:
        print(f"!! cumulative cost exceeded cap ${args.max_cost:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
