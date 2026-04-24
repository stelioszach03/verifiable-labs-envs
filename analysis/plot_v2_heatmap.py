#!/usr/bin/env python3
"""Render the v2 benchmark heatmap + markdown summary from results/llm_benchmark_v2.csv.

For each (model, env) pair, takes the reward of the highest-numbered
successful turn. Produces:
- results/benchmark_v2_heatmap.png
- results/benchmark_v2_summary.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
CSV = _REPO / "results" / "llm_benchmark_v2.csv"
PNG = _REPO / "results" / "benchmark_v2_heatmap.png"
MD = _REPO / "results" / "benchmark_v2_summary.md"

ENV_ORDER = (
    "sparse-fourier-recovery",
    "sparse-fourier-recovery-multiturn",
    "sparse-fourier-recovery-tools",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "lodopab-ct-simplified-multiturn",
)
ENV_LABEL = {
    "sparse-fourier-recovery": "SparseF",
    "sparse-fourier-recovery-multiturn": "SparseF-MT",
    "sparse-fourier-recovery-tools": "SparseF-Tools",
    "super-resolution-div2k-x4": "SuperRes",
    "lodopab-ct-simplified": "CT",
    "lodopab-ct-simplified-multiturn": "CT-MT",
}


def _load_final_turn_rewards() -> dict[tuple[str, str], list[float]]:
    """For each (model, env, seed), keep the highest-turn successful reward."""
    # (env, model, seed) -> (turn, reward)
    best: dict[tuple[str, str, int], tuple[int, float]] = {}
    with CSV.open() as fh:
        for row in csv.DictReader(fh):
            try:
                reward = float(row["reward"])
                turn = int(row["turn"])
            except (ValueError, TypeError):
                continue
            if row.get("parse_ok") != "True":
                continue
            key = (row["env"], row["model"], int(row["seed"]))
            if key not in best or turn > best[key][0]:
                best[key] = (turn, reward)

    pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (env_name, model, _seed), (_, reward) in best.items():
        pair[(env_name, model)].append(reward)
    return pair


def _build_matrix(pair: dict[tuple[str, str], list[float]]) -> tuple[np.ndarray, list[str], list[str]]:
    models = sorted({m for (_, m) in pair})
    envs_present = [e for e in ENV_ORDER if any((e, m) in pair for m in models)]
    mat = np.full((len(models), len(envs_present)), np.nan, dtype=np.float64)
    for i, m in enumerate(models):
        for j, e in enumerate(envs_present):
            vals = pair.get((e, m), [])
            if vals:
                mat[i, j] = fmean(vals)
    return mat, models, envs_present


def _plot(mat: np.ndarray, models: list[str], envs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 0.6 + 0.55 * len(models)))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(envs)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([ENV_LABEL.get(e, e) for e in envs], rotation=30, ha="right")
    ax.set_yticklabels([m.split("/")[-1] for m in models])
    for i in range(len(models)):
        for j in range(len(envs)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#555555")
            else:
                color = "white" if mat[i, j] < 0.45 else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title("v2 benchmark — mean reward by (model, env), final turn")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("reward")
    fig.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PNG}")


def _write_summary(mat: np.ndarray, models: list[str], envs: list[str]) -> None:
    lines = [
        "# v2 benchmark summary",
        "",
        "Source: [`results/llm_benchmark_v2.csv`](llm_benchmark_v2.csv). "
        "Each cell is the mean final-turn reward across seeds for one (model, env) pair. "
        "Dashes mean no successful row was recorded (either not attempted in this sweep or every seed parse-failed).",
        "",
        "## Table",
        "",
        "| model | " + " | ".join(ENV_LABEL.get(e, e) for e in envs) + " | mean |",
        "|---" + "|---:" * (len(envs) + 1) + "|",
    ]
    for i, m in enumerate(models):
        row_vals = mat[i]
        per_env = []
        for v in row_vals:
            per_env.append("—" if np.isnan(v) else f"{v:.3f}")
        valid = row_vals[~np.isnan(row_vals)]
        mean_s = "—" if valid.size == 0 else f"{valid.mean():.3f}"
        lines.append(f"| {m.split('/')[-1]} | " + " | ".join(per_env) + f" | **{mean_s}** |")

    # Column means
    col_means = []
    for j in range(len(envs)):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        col_means.append("—" if valid.size == 0 else f"{valid.mean():.3f}")
    lines.append("| **env mean** | " + " | ".join(col_means) + " | |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Multi-turn and tool-use envs report the LAST successful turn's reward. If every turn parse-failed, no row lands here. "
        "The full per-turn trajectory is in the CSV.\n"
        "- Multi-turn envs do not always help — see the per-row `turn_rewards` in `meta` for plateau / regression patterns.\n"
        "- The tool-use env relies on the adapter's `execute_tool_call` dispatch; a turn whose text is neither a valid final-answer JSON nor a valid tool call is recorded as a parse failure and ends the episode.\n"
        "- Opus 4.7 was dropped from this v2 sweep because Sprint-0 + Sprint-1 partial runs showed Sonnet ≈ Opus within noise on these envs; keeping it in would have blown the $3 cap per the plan's mitigation ladder."
    )
    MD.write_text("\n".join(lines))
    print(f"wrote {MD}")


def main() -> None:
    pair = _load_final_turn_rewards()
    if not pair:
        print("No rows found in CSV; nothing to plot.")
        return
    mat, models, envs = _build_matrix(pair)
    _plot(mat, models, envs)
    _write_summary(mat, models, envs)


if __name__ == "__main__":
    main()
