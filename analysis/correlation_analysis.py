#!/usr/bin/env python3
"""Cross-environment correlation analysis.

Reads `results/llm_benchmark.csv`, aggregates per-(model, env) mean reward on
successful calls, and computes the Spearman rank correlation matrix across
environment pairs — treating the models as observations.

Output:
- stdout: the 3x3 correlation matrix + the model x env mean-reward table.
- `results/env_correlation_matrix.csv`: machine-readable correlation matrix.

Interpretation:
- High correlation (> 0.7) means models ranked on one env tend to be ranked
  the same way on the other — the two envs measure overlapping capabilities.
- Low correlation (< 0.3) means the envs measure different things — useful
  as a battery.
- Spearman (rank-based) rather than Pearson (linear) so absolute-score
  differences across envs don't dominate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENVS = (
    "sparse-fourier-recovery",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
)


def _load_rewards(csv_path: Path) -> dict[tuple[str, str], list[float]]:
    """model, env -> list of successful rewards."""
    data: dict[tuple[str, str], list[float]] = defaultdict(list)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist", file=sys.stderr)
        sys.exit(2)
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = row["model"]
            env = row["env"]
            if env not in ENVS:
                continue
            if row.get("failure_mode"):
                continue
            try:
                reward = float(row["reward"])
            except (ValueError, TypeError):
                continue
            data[(model, env)].append(reward)
    return dict(data)


def _mean_table(rewards: dict[tuple[str, str], list[float]]) -> dict[str, dict[str, float]]:
    """model -> env -> mean reward (only for pairs with >=1 successful row)."""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, env), values in rewards.items():
        if not values:
            continue
        table[model][env] = fmean(values)
    return dict(table)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation using average ranks for ties."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None

    def _rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = sum((r - mean_rx) ** 2 for r in rx) ** 0.5
    denom_y = sum((r - mean_ry) ** 2 for r in ry) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "results" / "llm_benchmark.csv")
    parser.add_argument("--out", type=Path,
                        default=_PROJECT_ROOT / "results" / "env_correlation_matrix.csv")
    args = parser.parse_args()

    rewards = _load_rewards(args.csv)
    table = _mean_table(rewards)

    # ---- model x env mean-reward table ----
    models_with_all = sorted(m for m in table if all(e in table[m] for e in ENVS))
    print("Per-model mean reward (successful rows only):")
    header = f"  {'model':42s}  " + "  ".join(f"{e[:28]:>28s}" for e in ENVS) + "  " + f"{'mean':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in models_with_all:
        row = table[model]
        cells = "  ".join(f"{row[e]:>28.4f}" for e in ENVS)
        m_mean = fmean(row[e] for e in ENVS)
        print(f"  {model:42s}  {cells}  {m_mean:>8.4f}")

    # Models without a full triple are listed separately
    partial = sorted(m for m in table if m not in models_with_all)
    if partial:
        print("\nModels with incomplete env coverage (excluded from correlation):")
        for m in partial:
            missing = [e for e in ENVS if e not in table[m]]
            print(f"  {m}  missing: {missing}")

    # ---- correlation matrix ----
    print("\nSpearman rank correlation across env pairs"
          f" (n_models = {len(models_with_all)}):\n")
    matrix: dict[tuple[str, str], float | None] = {}
    for a, b in combinations(ENVS, 2):
        xs = [table[m][a] for m in models_with_all]
        ys = [table[m][b] for m in models_with_all]
        rho = _spearman(xs, ys)
        matrix[(a, b)] = rho

    # Print full 3x3
    print("  " + "".join(f"{e[:28]:>28s}" for e in ENVS))
    for i, env_i in enumerate(ENVS):
        row_cells = []
        for j, env_j in enumerate(ENVS):
            if i == j:
                row_cells.append(f"{1.00:>28.4f}")
            else:
                key = (env_i, env_j) if (env_i, env_j) in matrix else (env_j, env_i)
                rho = matrix.get(key)
                row_cells.append(f"{rho:>28.4f}" if rho is not None else f"{'n/a':>28s}")
        print(f"  {env_i[:26]:26s}" + "".join(row_cells))

    # ---- write CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["env_a", "env_b", "spearman_rho", "n_models"])
        for (a, b), rho in matrix.items():
            writer.writerow([a, b, rho if rho is not None else "", len(models_with_all)])
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
