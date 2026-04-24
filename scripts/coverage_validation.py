#!/usr/bin/env python3
"""Empirical conformal-coverage validation per env (N=100 fresh seeds).

For each of the 10 registered envs, runs the classical baseline on 100
fresh instances with seeds 20_000..20_099 (disjoint from train/eval seed
ranges used elsewhere), computes the per-entry conformal interval with
the env's deployed quantile, and records the empirical coverage (fraction
of true-support entries inside the [x_hat - q·sigma_hat, x_hat + q·sigma_hat]
band).

Output: results/coverage_validation.csv with columns
  (env, target_coverage, mean_coverage, std_coverage, n_samples)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifiable_labs_envs import load_environment  # noqa: E402


# Envs with a classical baseline + a run_baseline method.
ENVS = [
    "sparse-fourier-recovery",
    "sparse-fourier-recovery-multiturn",
    "sparse-fourier-recovery-tools",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "lodopab-ct-simplified-multiturn",
    "phase-retrieval",
    "phase-retrieval-multiturn",
    "mri-knee-reconstruction",
    "mri-knee-reconstruction-multiturn",
]


def main(n_samples: int = 100, seed_start: int = 20_000) -> None:
    rows = []
    for env_id in ENVS:
        try:
            env = load_environment(env_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipping {env_id}: {exc}", file=sys.stderr)
            continue
        target = 1.0 - float(getattr(env, "hyperparams", {}).get("alpha", 0.1))
        coverages = []
        for offset in range(n_samples):
            seed = seed_start + offset
            try:
                scored = env.run_baseline(seed=seed)
                cov = float(scored["meta"]["coverage"])
                coverages.append(cov)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {env_id} seed={seed} failed: {exc}",
                      file=sys.stderr)
                continue
        if not coverages:
            print(f"[warn] no coverages for {env_id}")
            continue
        arr = np.asarray(coverages)
        row = {
            "env": env_id,
            "target_coverage": round(target, 3),
            "mean_coverage": round(float(arr.mean()), 4),
            "std_coverage": round(float(arr.std()), 4),
            "min_coverage": round(float(arr.min()), 4),
            "max_coverage": round(float(arr.max()), 4),
            "n_samples": len(coverages),
        }
        rows.append(row)
        print(
            f"{env_id:38s} target={target:.3f}  "
            f"mean={row['mean_coverage']:.4f} "
            f"std={row['std_coverage']:.4f}  n={len(coverages)}"
        )

    out = REPO / "results" / "coverage_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["env", "target_coverage", "mean_coverage",
                        "std_coverage", "min_coverage", "max_coverage",
                        "n_samples"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    if rows:
        grand_mean = fmean(r["mean_coverage"] for r in rows)
        grand_std = pstdev([r["mean_coverage"] for r in rows]) if len(rows) > 1 else 0.0
        print(f"\nGrand mean coverage across {len(rows)} envs: "
              f"{grand_mean:.4f} ± {grand_std:.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n_samples=n)
