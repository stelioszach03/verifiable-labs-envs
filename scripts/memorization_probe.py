#!/usr/bin/env python3
"""Memorization probe for verifiable-labs-envs.

Empirical sanity check that a shipped LLM solver + environment pair actually
responds to its inputs. Two tests per model:

1. **Pipeline determinism** — run a fixed seed twice, compare rewards.
   LLMs at temperature=0 should produce near-identical output; large
   differences point to non-determinism in the transport layer.
2. **Cross-seed variance** — run ``n_seeds`` distinct seeds.
   ``std(reward) < 1e-2`` is flagged as a memorization / constant-output
   risk signal (the solver may not be responding to its input).

Writes one row per (model, env) pair to ``results/memorization_probe.csv``.

Budget: default run is 3 models x 1 env x 12 calls = 36 calls. At Sprint 0
per-call costs this is ~$0.03 total. Add ``--envs`` to widen at your own
cost risk.

Usage:
    python scripts/memorization_probe.py
    python scripts/memorization_probe.py --models openai/gpt-5.4-nano --n-seeds 5
    python scripts/memorization_probe.py --envs sparse-fourier-recovery,super-resolution-div2k-x4
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

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
from verifiable_labs_envs.solvers.llm_solver import get_adapter  # noqa: E402

DEFAULT_MODELS = (
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
)
DEFAULT_ENVS = ("sparse-fourier-recovery",)

DETERMINISM_TOLERANCE = 0.02  # |r_a - r_b| below this is "deterministic"
VARIANCE_FLAG_THRESHOLD = 1e-2  # std below this flags near-constant output

CSV_COLUMNS = [
    "timestamp",
    "model",
    "env",
    "n_seeds",
    "det_seed",
    "det_reward_a",
    "det_reward_b",
    "det_abs_diff",
    "det_ok",
    "var_seeds_start",
    "var_rewards_mean",
    "var_rewards_std",
    "var_rewards_min",
    "var_rewards_max",
    "var_flagged",
    "total_calls",
    "total_usd_cost",
    "total_failures",
]


def _score_one(solver: OpenRouterSolver, env_name: str, env, seed: int) -> tuple[float | None, float, str | None]:
    """Single call. Returns (reward or None, usd_cost, failure_message or None)."""
    adapter = get_adapter(env_name)
    instance = env.generate_instance(seed=seed)
    try:
        completion = solver.complete(adapter.system_prompt, adapter.build_user_prompt(instance))
    except LLMSolverError as exc:
        return None, 0.0, f"api:{exc}"
    cost = float(completion.usd_cost or 0.0)
    try:
        prediction = adapter.parse_response(completion.text, instance)
    except LLMSolverError as exc:
        return None, cost, f"parse:{exc}"
    scored = env.score(prediction, instance)
    return float(scored["reward"]), cost, None


def _probe_one(
    solver: OpenRouterSolver,
    env_name: str,
    env,
    *,
    det_seed: int,
    n_seeds: int,
    seeds_start: int,
) -> dict[str, object]:
    """Run determinism + variance tests for one (model, env) pair."""
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "timestamp": now,
        "model": solver.model,
        "env": env_name,
        "n_seeds": n_seeds,
        "det_seed": det_seed,
        "var_seeds_start": seeds_start,
        "total_calls": 0,
        "total_usd_cost": 0.0,
        "total_failures": 0,
    }

    # --- determinism test ---
    r_a, cost_a, fail_a = _score_one(solver, env_name, env, det_seed)
    row["total_calls"] += 1
    row["total_usd_cost"] = float(row["total_usd_cost"]) + cost_a
    r_b, cost_b, fail_b = _score_one(solver, env_name, env, det_seed)
    row["total_calls"] += 1
    row["total_usd_cost"] = float(row["total_usd_cost"]) + cost_b

    if r_a is None or r_b is None:
        row["det_reward_a"] = r_a
        row["det_reward_b"] = r_b
        row["det_abs_diff"] = None
        row["det_ok"] = False
        row["total_failures"] += int(r_a is None) + int(r_b is None)
    else:
        row["det_reward_a"] = round(r_a, 4)
        row["det_reward_b"] = round(r_b, 4)
        diff = abs(r_a - r_b)
        row["det_abs_diff"] = round(diff, 4)
        row["det_ok"] = diff <= DETERMINISM_TOLERANCE

    # --- variance test ---
    rewards: list[float] = []
    for offset in range(n_seeds):
        seed = seeds_start + offset
        r, cost, fail = _score_one(solver, env_name, env, seed)
        row["total_calls"] += 1
        row["total_usd_cost"] = float(row["total_usd_cost"]) + cost
        if r is None:
            row["total_failures"] += 1
        else:
            rewards.append(r)

    if len(rewards) >= 2:
        row["var_rewards_mean"] = round(statistics.fmean(rewards), 4)
        row["var_rewards_std"] = round(statistics.pstdev(rewards), 4)
        row["var_rewards_min"] = round(min(rewards), 4)
        row["var_rewards_max"] = round(max(rewards), 4)
        row["var_flagged"] = float(row["var_rewards_std"]) < VARIANCE_FLAG_THRESHOLD
    else:
        row["var_rewards_mean"] = None
        row["var_rewards_std"] = None
        row["var_rewards_min"] = None
        row["var_rewards_max"] = None
        row["var_flagged"] = True  # insufficient data -> conservatively flag

    row["total_usd_cost"] = round(float(row["total_usd_cost"]), 6)
    return row


def _print_row(row: dict[str, object]) -> None:
    head = f"{row['model']:40s}  {row['env']:28s}"
    det = f"det Δ={row['det_abs_diff']!s:>7} ({'OK' if row['det_ok'] else 'FAIL'})"
    var_std = row["var_rewards_std"]
    var_flag = "FLAG" if row["var_flagged"] else "OK"
    var = f"std={var_std!s:>7} ({var_flag})"
    cost = f"${float(row['total_usd_cost']):.4f}"
    fails = f"{row['total_failures']}/{row['total_calls']}"
    print(f"  {head}  |  {det}  |  {var}  |  {cost}  |  fails={fails}")


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="comma-separated OpenRouter model IDs")
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS),
                        help="comma-separated env names")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="number of seeds for variance test (default 10)")
    parser.add_argument("--seeds-start", type=int, default=0)
    parser.add_argument("--det-seed", type=int, default=42)
    parser.add_argument("--max-cost", type=float, default=1.0,
                        help="hard cumulative USD cap (default 1.0)")
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "results" / "memorization_probe.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    envs = tuple(e.strip() for e in args.envs.split(",") if e.strip())

    calls_per_pair = 2 + args.n_seeds
    total_calls = len(models) * len(envs) * calls_per_pair
    print(f"Planned probe: {len(models)} model(s) x {len(envs)} env(s) x "
          f"({calls_per_pair} calls) = {total_calls} calls")
    print(f"Models: {list(models)}")
    print(f"Envs:   {list(envs)}")
    print(f"Max cost: ${args.max_cost:.2f}\n")

    if args.dry_run:
        return 0
    if not HAS_OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set. Load .env or export the key.", file=sys.stderr)
        return 2

    env_cache: dict[str, object] = {name: load_environment(name) for name in envs}
    cumulative = 0.0
    rows: list[dict[str, object]] = []

    for model in models:
        print(f"=== {model} ===")
        solver = OpenRouterSolver(model=model, max_tokens=4096, timeout_s=120)
        for env_name in envs:
            env = env_cache[env_name]
            row = _probe_one(
                solver, env_name, env,
                det_seed=args.det_seed,
                n_seeds=args.n_seeds,
                seeds_start=args.seeds_start,
            )
            rows.append(row)
            _print_row(row)
            cumulative += float(row["total_usd_cost"])
            if cumulative > args.max_cost:
                print(f"\nABORT: cumulative cost ${cumulative:.4f} exceeds cap "
                      f"${args.max_cost:.2f}.", file=sys.stderr)
                _write_csv(rows, args.csv)
                return 3

    _write_csv(rows, args.csv)
    print(f"\nCumulative probe cost: ${cumulative:.4f}")
    print(f"CSV appended: {args.csv}")

    any_flagged = any(r.get("var_flagged") for r in rows) or any(not r.get("det_ok") for r in rows)
    if any_flagged:
        print("NOTE: at least one row flagged (determinism failure or low variance). "
              "Inspect the CSV before drawing conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
