#!/usr/bin/env python3
"""LLM benchmark for verifiable-labs-envs.

Runs one or more LLM models (via OpenRouter) against the three environments.
Records reward components, failure modes, latency, token usage, and per-call
USD cost. Appends rows to ``results/llm_benchmark.csv`` and overwrites
``results/llm_benchmark.md`` with a pretty aggregate table.

Presets:
  smoke          — free-tier only, 1 instance per env, 0 USD
  paid-small     — cheap+mid paid models, 3 instances per env, ~2 USD
  paid-full      — all paid models, 5 instances per env, ~8 USD

Safety:
  --max-cost USD   hard-stops the sweep as soon as cumulative cost exceeds this
  --per-call-cap USD  aborts after any single call costs more than this
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

# Bootstrap src/ onto sys.path before importing the package — the hatchling
# editable install sometimes writes a .pth without a terminating newline,
# which Python's site module silently drops. Belt-and-braces.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from verifiable_labs_envs import load_environment  # noqa: E402
from verifiable_labs_envs.solvers import (  # noqa: E402
    HAS_OPENROUTER_KEY,
    LLMSolverError,
    OpenRouterSolver,
)

ALL_ENVS = (
    "sparse-fourier-recovery",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
)

PRESETS = {
    "smoke": {
        "models": (
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat-v3:free",
        ),
        "n_instances": 1,
    },
    "paid-small": {
        "models": (
            "openai/gpt-5.4-nano",
            "anthropic/claude-haiku-4.5",
            "openai/gpt-5.4-mini",
            "google/gemini-3.1-pro",
            "anthropic/claude-sonnet-4.6",
        ),
        "n_instances": 3,
    },
    "paid-full": {
        "models": (
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-5.4-nano",
            "anthropic/claude-haiku-4.5",
            "openai/gpt-5.4-mini",
            "google/gemini-3.1-pro",
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.4",
            "anthropic/claude-opus-4.7",
        ),
        "n_instances": 5,
    },
}

CSV_COLUMNS = [
    "timestamp",
    "model",
    "env",
    "seed",
    "reward",
    "components",
    "failure_mode",
    "failure_message",
    "latency_s",
    "usd_cost",
    "prompt_tokens",
    "completion_tokens",
    "reported_model",
]


def _write_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def _load_envs(env_names: list[str]) -> dict[str, object]:
    envs: dict[str, object] = {}
    for name in env_names:
        envs[name] = load_environment(name)
    return envs


def _run_one(
    solver: OpenRouterSolver,
    env_name: str,
    env: object,
    seed: int,
) -> dict[str, object]:
    """Single (model, env, seed) call. Returns a row dict for CSV + summary."""
    now = datetime.now(UTC).isoformat()
    instance = env.generate_instance(seed=seed)  # type: ignore[attr-defined]
    row: dict[str, object] = {
        "timestamp": now,
        "model": solver.model,
        "env": env_name,
        "seed": seed,
        "reward": None,
        "components": "",
        "failure_mode": "",
        "failure_message": "",
        "latency_s": 0.0,
        "usd_cost": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_model": "",
    }

    # Build prompt + call the model directly so we can record transport metadata
    # even when adapter parsing later fails.
    from verifiable_labs_envs.solvers.llm_solver import get_adapter

    adapter = get_adapter(env_name)
    try:
        completion = solver.complete(
            adapter.system_prompt,
            adapter.build_user_prompt(instance),
        )
    except LLMSolverError as exc:
        row["failure_mode"] = "api"
        row["failure_message"] = str(exc)
        return row

    row["latency_s"] = round(completion.latency_s, 3)
    row["usd_cost"] = completion.usd_cost
    row["prompt_tokens"] = completion.prompt_tokens
    row["completion_tokens"] = completion.completion_tokens
    row["reported_model"] = completion.model

    try:
        prediction = adapter.parse_response(completion.text, instance)
    except LLMSolverError as exc:
        row["failure_mode"] = "parse"
        row["failure_message"] = str(exc)
        return row
    except Exception as exc:  # noqa: BLE001 — benchmark is resilience-first
        row["failure_mode"] = "unexpected"
        row["failure_message"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        scored = env.score(prediction, instance)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        row["failure_mode"] = "score"
        row["failure_message"] = f"{type(exc).__name__}: {exc}"
        return row

    row["reward"] = round(float(scored["reward"]), 4)
    components = {k: round(float(v), 4) for k, v in scored["components"].items()}
    row["components"] = ", ".join(f"{k}={v:.3f}" for k, v in components.items())
    return row


def _format_row_line(row: dict[str, object]) -> str:
    fields = [
        f"{row['model']:40s}",
        f"{row['env']:28s}",
        f"seed={row['seed']}",
    ]
    if row["failure_mode"]:
        fields.append(f"FAIL({row['failure_mode']}): {row['failure_message']}")
    else:
        fields.append(f"reward={row['reward']:.3f}  {row['components']}")
    cost = row["usd_cost"]
    if cost is not None:
        fields.append(f"${float(cost):.4f}")
    fields.append(f"{float(row['latency_s']):.2f}s")
    return "  |  ".join(fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated list of OpenRouter model IDs (overrides --preset)")
    parser.add_argument("--envs", type=str, default=",".join(ALL_ENVS))
    parser.add_argument("--n", "--n-instances", dest="n_instances", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-cost", type=float, default=12.0,
                        help="hard cumulative USD cap (default 12).")
    parser.add_argument("--per-call-cap", type=float, default=0.50,
                        help="abort if any single call costs more than this (default 0.50).")
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "results" / "llm_benchmark.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned sweep without calling the API.")
    args = parser.parse_args()

    if args.models:
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        n_instances = args.n_instances or 1
    elif args.preset:
        preset = PRESETS[args.preset]
        models = preset["models"]
        n_instances = args.n_instances or preset["n_instances"]
    else:
        parser.error("one of --preset or --models is required")

    env_names = [e.strip() for e in args.envs.split(",") if e.strip()]
    for env_name in env_names:
        if env_name not in ALL_ENVS:
            parser.error(f"unknown env '{env_name}'. choose from: {ALL_ENVS}")

    total_calls = len(models) * len(env_names) * n_instances
    print(f"Planned sweep: {len(models)} model(s) × {len(env_names)} env(s) × "
          f"{n_instances} instance(s) = {total_calls} calls")
    print(f"Models: {list(models)}")
    print(f"Envs:   {env_names}")
    print(f"Max cost: ${args.max_cost:.2f}   per-call cap: ${args.per_call_cap:.2f}")
    print()

    if args.dry_run:
        return 0

    if not HAS_OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set. Load .env or export the key.", file=sys.stderr)
        return 2

    envs = _load_envs(env_names)

    cumulative_cost: float = 0.0
    total_failures = 0
    for model in models:
        print(f"\n=== {model} ===")
        solver = OpenRouterSolver(model=model, max_tokens=4096, timeout_s=180)
        for env_name in env_names:
            env = envs[env_name]
            for offset in range(n_instances):
                seed = args.seed_start + offset
                row = _run_one(solver, env_name, env, seed)
                _write_csv_row(args.csv, row)
                print("  " + _format_row_line(row))
                if row["failure_mode"]:
                    total_failures += 1
                if row["usd_cost"] is not None:
                    call_cost = float(row["usd_cost"])
                    cumulative_cost += call_cost
                    if call_cost > args.per_call_cap:
                        print(f"\nABORT: single call cost ${call_cost:.4f} exceeds per-call cap "
                              f"${args.per_call_cap:.2f}.", file=sys.stderr)
                        return 3
                    if cumulative_cost > args.max_cost:
                        print(f"\nABORT: cumulative cost ${cumulative_cost:.4f} exceeds "
                              f"max ${args.max_cost:.2f}.", file=sys.stderr)
                        return 4

    print(f"\nDone. Cumulative reported cost: ${cumulative_cost:.4f}  "
          f"failures: {total_failures}/{total_calls}")
    print(f"CSV appended: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
