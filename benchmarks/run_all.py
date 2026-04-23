#!/usr/bin/env python3
"""Run every environment's reference baseline and zero baseline over a
small seed sweep, then print a consolidated markdown-ready table. This is
what gets linked from the README and what ships in `baselines/<env>/`
directories on release.

Usage::

    python benchmarks/run_all.py                 # default: seeds 0..4 per env, fast cal
    python benchmarks/run_all.py --seeds 10      # broader sweep
    python benchmarks/run_all.py --full          # full calibration (slower)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs import lodopab_ct as ct
from verifiable_labs_envs.envs import sparse_fourier as sf
from verifiable_labs_envs.envs import super_resolution as sr

ZERO_BASELINES = {
    "sparse-fourier-recovery": sf.zero_baseline,
    "super-resolution-div2k-x4": sr.zero_baseline,
    "lodopab-ct-simplified": ct.zero_baseline,
}


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _run_env(env_name: str, n_seeds: int, fast: bool) -> dict[str, object]:
    env = load_environment(env_name)  # fast=True by default where applicable
    zero_fn = ZERO_BASELINES[env_name]

    baseline_rewards: list[float] = []
    zero_rewards: list[float] = []
    all_components: dict[str, list[float]] = {}

    for seed in range(n_seeds):
        # Reference baseline
        out_ref = env.run_baseline(seed=seed)
        baseline_rewards.append(out_ref["reward"])
        for name, value in out_ref["components"].items():
            all_components.setdefault(name, []).append(value)

        # Zero baseline
        inst = env.generate_instance(seed=seed)
        zero_pred = zero_fn(**inst.as_inputs())
        zero_rewards.append(env.score(zero_pred, inst)["reward"])

    return {
        "env": env_name,
        "n_seeds": n_seeds,
        "fast_calibration": fast,
        "conformal_quantile": env.conformal_quantile,
        "baseline_reward": _summary(baseline_rewards),
        "zero_reward": _summary(zero_rewards),
        "baseline_components": {k: _summary(v) for k, v in all_components.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="use full (not fast) calibration")
    parser.add_argument("--out", type=Path, default=None, help="write JSON results here")
    args = parser.parse_args()

    rows = []
    for env_name in sorted(list_environments()):
        print(f"running {env_name} ...", flush=True)
        row = _run_env(env_name, n_seeds=args.seeds, fast=not args.full)
        rows.append(row)

    print()
    print("| environment | reference reward | zero reward | gap | conformal q |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        ref = row["baseline_reward"]["mean"]
        zero = row["zero_reward"]["mean"]
        gap = ref - zero
        q = row["conformal_quantile"]
        print(f"| `{row['env']}` | {ref:.3f} | {zero:.3f} | +{gap:.3f} | {q:.3f} |")

    print()
    print("Component breakdown (reference baseline, mean over seeds):")
    for row in rows:
        parts = ", ".join(
            f"{name}={info['mean']:.3f}"
            for name, info in row["baseline_components"].items()
        )
        print(f"  {row['env']:30s} {parts}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
