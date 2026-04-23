#!/usr/bin/env python3
"""Plot the env correlation heatmap and the per-model mean-score bar chart.

Reads:
- `results/llm_benchmark.csv` to aggregate per-(model, env) mean reward.
- `results/env_correlation_matrix.csv` for the Spearman correlations.

Writes:
- `results/env_correlation_heatmap.png`
- `results/model_mean_scores.png`

Both PNGs are referenced by `README.md` and embedded in the static HF
Spaces leaderboard planned for Phase 7.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENVS = (
    "sparse-fourier-recovery",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
)
ENV_LABEL = {
    "sparse-fourier-recovery": "SparseF",
    "super-resolution-div2k-x4": "SuperRes",
    "lodopab-ct-simplified": "LoDoPaB-CT",
}

CLASSICAL_BASELINE = {
    "sparse-fourier-recovery": 0.869,
    "super-resolution-div2k-x4": 0.629,
    "lodopab-ct-simplified": 0.712,
}
ZERO_BASELINE = {
    "sparse-fourier-recovery": 0.336,
    "super-resolution-div2k-x4": 0.425,
    "lodopab-ct-simplified": 0.151,
}


def _load_correlation_matrix(path: Path) -> np.ndarray:
    """Build a 3x3 symmetric matrix from the pairwise CSV rows."""
    mat = np.ones((len(ENVS), len(ENVS)), dtype=np.float64)
    idx = {env: i for i, env in enumerate(ENVS)}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            a, b = row["env_a"], row["env_b"]
            if a not in idx or b not in idx:
                continue
            rho = row["spearman_rho"]
            if rho == "" or rho is None:
                continue
            mat[idx[a], idx[b]] = float(rho)
            mat[idx[b], idx[a]] = float(rho)
    return mat


def _plot_heatmap(mat: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    labels = [ENV_LABEL[e] for e in ENVS]
    ax.set_xticks(range(len(ENVS)))
    ax.set_yticks(range(len(ENVS)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(ENVS)):
        for j in range(len(ENVS)):
            color = "white" if abs(mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", color=color)
    ax.set_title("Cross-env Spearman rank correlation (n=6 models)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _load_model_means(csv_path: Path) -> dict[str, dict[str, float]]:
    rewards: dict[tuple[str, str], list[float]] = defaultdict(list)
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["env"] not in ENVS:
                continue
            if row.get("failure_mode"):
                continue
            try:
                rewards[(row["model"], row["env"])].append(float(row["reward"]))
            except (ValueError, TypeError):
                continue
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (model, env), vals in rewards.items():
        if vals:
            out[model][env] = fmean(vals)
    return dict(out)


def _plot_model_means(model_means: dict[str, dict[str, float]], out_path: Path) -> None:
    # Keep only models with coverage on all 3 envs for a fair comparison.
    models = sorted(m for m in model_means if all(e in model_means[m] for e in ENVS))
    means = [fmean(model_means[m][e] for e in ENVS) for m in models]

    # Prepend a classical-baseline bar; append zero-baseline.
    classical_mean = fmean(CLASSICAL_BASELINE.values())
    zero_mean = fmean(ZERO_BASELINE.values())
    display_labels = ["Classical baseline", *models, "Zero baseline"]
    display_values = [classical_mean, *means, zero_mean]

    # Short human labels for the bars (strip org prefix if present).
    short = [lbl.split("/")[-1] for lbl in display_labels]
    short[0] = display_labels[0]
    short[-1] = display_labels[-1]

    colors = ["#1b7f3a"] + ["#3872b0"] * len(models) + ["#888888"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(range(len(display_values)), display_values, color=colors)
    ax.set_xticks(range(len(display_values)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_ylabel("Mean reward (3 envs)")
    ax.set_ylim(0, 1)
    ax.set_title("Mean reward by model (averaged across 3 envs)")
    ax.axhline(classical_mean, color="#1b7f3a", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(zero_mean, color="#888888", lw=0.8, ls="--", alpha=0.6)
    for bar, val in zip(bars, display_values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    csv_path = _PROJECT_ROOT / "results" / "llm_benchmark.csv"
    corr_csv = _PROJECT_ROOT / "results" / "env_correlation_matrix.csv"
    if not corr_csv.exists():
        print(f"ERROR: {corr_csv} missing. Run `python analysis/correlation_analysis.py` first.",
              file=sys.stderr)
        return 2

    mat = _load_correlation_matrix(corr_csv)
    _plot_heatmap(mat, _PROJECT_ROOT / "results" / "env_correlation_heatmap.png")

    model_means = _load_model_means(csv_path)
    _plot_model_means(model_means, _PROJECT_ROOT / "results" / "model_mean_scores.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
