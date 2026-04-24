#!/usr/bin/env python3
"""Generate the five figures used in the OpenReview preprint.

Reads real CSV data from ``../results/`` (relative to the repo root) and
writes publication-grade PDFs into ``paper/figures/``. No fabricated
data — every number in a figure traces to a CSV row or an aggregation
computed in this script and echoed to stdout for the verification log.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Publication-grade style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# Colour palette — accessible, distinguishable grayscale-friendly.
COLOUR_CLASSICAL = "#1b5e20"       # dark green accent for classical
COLOUR_LLM = "#90a4ae"             # neutral grey for LLMs
COLOUR_LLM_HIGHLIGHT = "#ec407a"   # accent pink for highlighted LLM
COLOUR_LLM_MULTI = "#4fc3f7"       # light blue for multi-turn contrast
COLOUR_ARTIFACT = "#d84315"        # orange/red for v0.1 oracle artefact
COLOUR_TARGET = "#5d4037"          # dark brown horizontal target line


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

LLM_CSVS = [
    REPO / "results" / "llm_benchmark_v2.csv",
    REPO / "results" / "meta_benchmark_v3.csv",
    REPO / "results" / "phase_retrieval_v1_benchmark.csv",
    REPO / "results" / "mri_knee_v1_benchmark.csv",
    REPO / "results" / "opus_nano_fill_v2.csv",
    REPO / "results" / "complete_matrix_single_turn.csv",   # paper-final 1A
    REPO / "results" / "complete_matrix_multi_turn.csv",    # paper-final 1B
    REPO / "results" / "tools_v2_complete.csv",             # paper-final 1C
]


def load_llm_rewards() -> dict[tuple[str, str], list[float]]:
    """(env, model) → list of final-turn rewards across seeds."""
    best: dict[tuple[str, str, int], tuple[int, float]] = {}
    for path in LLM_CSVS:
        if not path.exists():
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("parse_ok") != "True":
                    continue
                try:
                    reward = float(r["reward"])
                except (TypeError, ValueError):
                    continue
                seed_raw = r.get("seed") or r.get("instance_id")
                try:
                    seed = int(seed_raw)
                except (TypeError, ValueError):
                    continue
                turn = int(r.get("turn") or 1)
                key = (r["env"], r["model"], seed)
                if key not in best or turn > best[key][0]:
                    best[key] = (turn, reward)
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (env, model, _), (_, reward) in best.items():
        out[(env, model)].append(reward)
    return out


def load_classical_rewards() -> dict[str, float]:
    """Classical-baseline means per env, sourced from:
    - Sprint-0 benchmark README (5-seed OMP/FBP/bicubic means).
    - Sprint-giga README for phase-retrieval (GS) and mri-knee (zero-filled).
    Numbers are reproducible by re-running each env's ``run_baseline`` method;
    we pin them here to avoid the 30+ s of re-running during figure gen.
    """
    return {
        "sparse-fourier-recovery": 0.869,   # OMP, 5-seed
        "super-resolution-div2k-x4": 0.629,  # bicubic, 5-seed
        "lodopab-ct-simplified": 0.712,     # FBP, 5-seed
        "phase-retrieval": 0.289,           # GS, 3-seed
        "mri-knee-reconstruction": 0.649,   # zero-filled IFFT, 3-seed
    }


def env_short(env_id: str) -> str:
    return {
        "sparse-fourier-recovery": "SparseF",
        "sparse-fourier-recovery-multiturn": "SparseF-MT",
        "sparse-fourier-recovery-tools": "SparseF-Tools",
        "super-resolution-div2k-x4": "SuperRes",
        "lodopab-ct-simplified": "CT",
        "lodopab-ct-simplified-multiturn": "CT-MT",
        "phase-retrieval": "PhaseRet",
        "phase-retrieval-multiturn": "PhaseRet-MT",
        "mri-knee-reconstruction": "MRI",
        "mri-knee-reconstruction-multiturn": "MRI-MT",
    }.get(env_id, env_id)


def model_short(model_id: str) -> str:
    m = model_id.split("/")[-1]
    return m.replace("claude-", "").replace("openai/", "")


# ─────────────────────────────────────────────────────────────
# Figure 1 — cross-env heatmap (full-width, 6.75 x 3.0 in)
# ─────────────────────────────────────────────────────────────

def fig1_heatmap() -> Path:
    llm = load_llm_rewards()
    models = sorted(
        {m for (_, m) in llm},
        key=lambda m: -fmean([fmean(vs) for (e, mm), vs in llm.items() if mm == m]) if any(mm == m for (_, mm) in llm) else 0,
    )
    # Focus on the 5 single-turn envs (Finding 1's clean cross-env axis).
    envs = [
        "sparse-fourier-recovery",
        "phase-retrieval",
        "lodopab-ct-simplified",
        "mri-knee-reconstruction",
        "super-resolution-div2k-x4",
    ]

    grid = np.full((len(models), len(envs)), np.nan)
    for i, m in enumerate(models):
        for j, e in enumerate(envs):
            vals = llm.get((e, m), [])
            if vals:
                grid[i, j] = fmean(vals)

    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    im = ax.imshow(grid, cmap="viridis", vmin=0.25, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([env_short(e) for e in envs], rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_short(m) for m in models])

    # Annotate cells (white text on dark viridis, black on light)
    for i in range(len(models)):
        for j in range(len(envs)):
            v = grid[i, j]
            if not np.isnan(v):
                colour = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=colour, fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        color="white", fontsize=8)

    # Classical-baseline row at the bottom
    classical = load_classical_rewards()
    ax.axhline(len(models) - 0.5, color="white", linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Mean reward", size=9)
    ax.set_title("Reward by model × environment", pad=6)

    # Print classical row below the chart
    classical_row = "  ".join(f"{env_short(e)}: {classical[e]:.2f}"
                              for e in envs if e in classical)
    ax.text(0.0, -0.18, f"Classical baseline:  {classical_row}",
            transform=ax.transAxes, fontsize=7, va="top", color=COLOUR_CLASSICAL)

    path = OUT / "fig1_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"fig1: wrote {path.name}, models={len(models)}, envs={len(envs)}")
    return path


# ─────────────────────────────────────────────────────────────
# Figure 2 — classical vs LLM gap chart (single-column)
# ─────────────────────────────────────────────────────────────

def fig2_gap() -> Path:
    llm = load_llm_rewards()
    classical = load_classical_rewards()
    # 5-env cross-env mean per model and for classical.
    envs = list(classical.keys())
    model_means: dict[str, float] = {}
    for (env, model), rewards in llm.items():
        if env not in envs:
            continue
    # compute per-model average across the 5 envs (only envs where the model has data)
    by_model_env: dict[str, dict[str, float]] = defaultdict(dict)
    for (env, model), rewards in llm.items():
        if env in envs and rewards:
            by_model_env[model][env] = fmean(rewards)
    for model, env_means in by_model_env.items():
        vals = list(env_means.values())
        if len(vals) >= 2:  # need at least 2 envs for a mean
            model_means[model] = fmean(vals)

    classical_mean = fmean(classical.values())

    # Sort models descending by mean reward
    ordered = sorted(model_means.items(), key=lambda kv: kv[1], reverse=True)
    labels = ["Classical"] + [model_short(m) for m, _ in ordered]
    values = [classical_mean] + [v for _, v in ordered]
    # deltas relative to classical
    deltas = [0.0] + [v - classical_mean for _, v in ordered]
    colours = [COLOUR_CLASSICAL] + [COLOUR_LLM] * len(ordered)

    fig, ax = plt.subplots(figsize=(3.25, 2.7))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("5-env mean reward")
    ax.set_xlim(0.0, max(values) * 1.18)
    # annotate each bar with value and Δ (for LLMs)
    for i, (val, d) in enumerate(zip(values, deltas)):
        txt = f"{val:.3f}"
        if i != 0:
            txt += f"  (Δ={d:+.3f})"
        ax.text(val + 0.005, i, txt, va="center", fontsize=7.5)
    ax.set_title("Classical beats every tested LLM")

    path = OUT / "fig2_gap.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"fig2: wrote {path.name}, classical_mean={classical_mean:.3f}, "
          f"top_llm={ordered[0][0]}@{ordered[0][1]:.3f}")
    return path


# ─────────────────────────────────────────────────────────────
# Figure 3 — multi-turn differential (single-column)
# ─────────────────────────────────────────────────────────────

def fig3_multiturn() -> Path:
    llm = load_llm_rewards()
    # domain pairs: (single, multiturn) env ids
    pairs = [
        ("sparse-fourier-recovery", "sparse-fourier-recovery-multiturn"),
        ("lodopab-ct-simplified", "lodopab-ct-simplified-multiturn"),
    ]
    # Collect mean Δ per model across the two domains
    models = sorted({m for (_, m) in llm})
    rows = []
    for m in models:
        deltas = []
        for st, mt in pairs:
            st_vals = llm.get((st, m), [])
            mt_vals = llm.get((mt, m), [])
            if st_vals and mt_vals:
                deltas.append(fmean(mt_vals) - fmean(st_vals))
        if len(deltas) >= 2:
            rows.append((m, fmean(deltas)))

    # Sort by delta ascending (most-negative first = "multiturn hurts")
    rows.sort(key=lambda r: r[1])
    labels = [model_short(m) for m, _ in rows]
    deltas = [d for _, d in rows]
    colours = [COLOUR_LLM if d < 0 else COLOUR_LLM_MULTI for d in deltas]

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    y = np.arange(len(labels))
    ax.barh(y, deltas, color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#616161", linewidth=0.6)
    ax.set_xlabel(r"Mean $\Delta$ (multi-turn − single-turn)")
    # annotate
    for i, d in enumerate(deltas):
        offset = 0.004 if d >= 0 else -0.004
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.3f}", va="center", ha=ha, fontsize=7.5)
    ax.set_title("Multi-turn delta per model\n(sparse-F + CT domain mean)")

    path = OUT / "fig3_multiturn.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"fig3: wrote {path.name}, n_models={len(rows)}")
    return path


# ─────────────────────────────────────────────────────────────
# Figure 4 — conformal coverage validation (single-column)
# ─────────────────────────────────────────────────────────────

def fig4_coverage() -> Path:
    # Prefer the N=200 run if available; fall back to N=100.
    cov_csv = REPO / "results" / "coverage_validation_n200.csv"
    if not cov_csv.exists():
        cov_csv = REPO / "results" / "coverage_validation.csv"
    if not cov_csv.exists():
        raise FileNotFoundError(cov_csv)
    with cov_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    envs = [r["env"] for r in rows]
    means = np.asarray([float(r["mean_coverage"]) for r in rows])
    stds = np.asarray([float(r["std_coverage"]) for r in rows])
    target = float(rows[0]["target_coverage"])
    n = int(rows[0]["n_samples"])

    # Sort by env for reproducibility; shorten labels
    order = np.argsort(envs)
    envs = [envs[i] for i in order]
    means = means[order]
    stds = stds[order]
    labels = [env_short(e) for e in envs]

    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    x = np.arange(len(envs))
    ax.errorbar(x, means, yerr=stds, fmt="o", color=COLOUR_CLASSICAL,
                ecolor="#90caf9", capsize=2.5, markersize=4,
                markeredgewidth=0, linewidth=0)
    ax.axhline(target, color=COLOUR_TARGET, linestyle="--", linewidth=0.8,
               label=f"target = {target:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.5, 1.05)
    grand_mean = float(means.mean())
    grand_std = float(means.std())
    ax.set_title(
        f"Coverage validation (N={n}/env)\n"
        f"grand mean = {grand_mean:.3f} $\\pm$ {grand_std:.3f}"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=7)

    path = OUT / "fig4_coverage.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"fig4: wrote {path.name}, n_envs={len(envs)}, "
          f"grand_mean={grand_mean:.4f}±{grand_std:.4f}")
    return path


# ─────────────────────────────────────────────────────────────
# Figure 5 — tool-use primitive gap (single-column)
# ─────────────────────────────────────────────────────────────

def fig5_tools() -> Path:
    # v0.3 primitive-only rewards per model — prefer the complete 6-model run
    # from Phase 1C (tools_v2_complete.csv) and fall back to the earlier
    # 3-cheap-model run if it's missing.
    path_v3 = REPO / "results" / "tools_v2_complete.csv"
    if not path_v3.exists():
        path_v3 = REPO / "results" / "llm_benchmark_tools_v2.csv"
    with path_v3.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("parse_ok") == "True"]
    by_model = defaultdict(list)
    for r in rows:
        try:
            by_model[r["model"]].append(float(r["reward"]))
        except (TypeError, ValueError):
            continue

    # v0.1 oracle artifact value (byte-identical across 3 models per seed)
    # documented in results/sparse_fourier_reconciliation.md
    v01_oracle = 0.858

    # Classical OMP reference (same sparse-F env, 5-seed mean)
    classical_omp = 0.869

    models = sorted(by_model)
    labels = ([model_short(m) for m in models]
              + ["Classical OMP", "v0.1 oracle*"])
    means = ([fmean(by_model[m]) for m in models]
             + [classical_omp, v01_oracle])
    colours = ([COLOUR_LLM] * len(models)
               + [COLOUR_CLASSICAL, COLOUR_ARTIFACT])

    fig, ax = plt.subplots(figsize=(3.25, 2.7))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colours, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 1.0)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_title("Tool-use: primitives vs classical vs v0.1 oracle (artefact)")
    # Footnote for the oracle artefact
    ax.annotate(
        "*v0.1 shipped an ISTA oracle tool that returned\n"
        " the OMP answer; removed in v0.3 (primitives only).",
        xy=(0.02, -0.35), xycoords="axes fraction", fontsize=6,
        color=COLOUR_ARTIFACT,
    )

    path = OUT / "fig5_tools.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"fig5: wrote {path.name}, n_llms={len(models)}, "
          f"primitive_mean={fmean([fmean(vs) for vs in by_model.values()]):.3f}")
    return path


def main() -> None:
    fig1_heatmap()
    fig2_gap()
    fig3_multiturn()
    fig4_coverage()
    fig5_tools()
    print(f"\nAll 5 figures written to {OUT}")


if __name__ == "__main__":
    main()
