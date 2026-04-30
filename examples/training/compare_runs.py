"""Paired pre/post comparison + plots for the M7 GRPO writeup.

Inputs (Drive paths):
  qwen15b_base_eval.jsonl           BEFORE  (M5)
  qwen15b_grpo_eval_ckpt100.jsonl   AFTER-100   (M7)
  qwen15b_grpo_eval_ckpt250.jsonl   AFTER-250   (M7)
  qwen15b_grpo_eval_ckpt500.jsonl   AFTER-500   (M7)

Per JSONL we expect 100 seeds × 3 samples = 300 rows. Each row carries
``metadata.instance_hash`` for paired comparison (pairing on
``instance_hash`` rather than seed defends against silent env drift).

Outputs (Drive):
  comparison_stats.json
  reward_curve.png
  before_after_distribution.png
  component_deltas.png
  trajectory.png
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Headless plotting on the remote.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


DEFAULT_OUTDIR = Path("/content/drive/MyDrive/verifiable-labs/training_outputs")
DEFAULT_PLOTS_DIR = Path("/content/drive/MyDrive/verifiable-labs/results_remote")
DEFAULT_BASELINE = DEFAULT_OUTDIR / "qwen15b_base_eval.jsonl"
DEFAULT_TRAIN_LOG = Path(
    "/content/drive/MyDrive/verifiable-labs/checkpoints/qwen15b_grpo_sf_v1/training_log.jsonl"
)


# ── data loading ──────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_seed_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group rows by instance_hash; return dict keyed by instance_hash with
    averaged reward + components across the per-seed samples."""
    by_inst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ih = r.get("metadata", {}).get("instance_hash") or r.get("instance_hash")
        if not ih:
            continue
        by_inst[ih].append(r)

    out: dict[str, dict[str, Any]] = {}
    for ih, group in by_inst.items():
        rewards = [g["reward"] for g in group]
        components = ["nmse", "support", "conformal", "parse_valid", "format_valid"]
        comp_means = {
            c: statistics.fmean(g["reward_components"].get(c, 0.0) for g in group)
            for c in components
        }
        out[ih] = {
            "n_samples": len(group),
            "reward_mean": statistics.fmean(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "components": comp_means,
            "seed": group[0].get("seed"),
        }
    return out


# ── stats ─────────────────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    lo = float(np.percentile(means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def cohen_d_paired(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return 0.0
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0
    return mean / sd


def paired_compare(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Pair on instance_hash, compute Δ stats."""
    common = sorted(set(before) & set(after))
    n = len(common)
    if n == 0:
        return {"error": "no overlapping instance_hash between BEFORE and AFTER",
                "n_overlap": 0}

    b_rewards = np.array([before[k]["reward_mean"] for k in common], dtype=np.float64)
    a_rewards = np.array([after[k]["reward_mean"] for k in common], dtype=np.float64)
    diffs = a_rewards - b_rewards

    mean_before = float(np.mean(b_rewards))
    mean_after = float(np.mean(a_rewards))
    delta_mean = float(np.mean(diffs))
    delta_std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(diffs)

    if n >= 5 and not np.allclose(diffs, 0.0):
        try:
            wilcoxon = scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except Exception as e:  # noqa: BLE001
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
            print(f"WARN: wilcoxon failed: {e}")
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    cd = cohen_d_paired(diffs)

    # Component-wise paired Δ.
    comp_keys = ["nmse", "support", "conformal", "parse_valid", "format_valid"]
    comp_deltas: dict[str, float] = {}
    for c in comp_keys:
        b_vals = np.array([before[k]["components"].get(c, 0.0) for k in common])
        a_vals = np.array([after[k]["components"].get(c, 0.0) for k in common])
        comp_deltas[c] = float(np.mean(a_vals - b_vals))

    return {
        "n_overlap": n,
        "mean_reward_before": mean_before,
        "mean_reward_after": mean_after,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_ci95": [ci_lo, ci_hi],
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "cohen_d": cd,
        "component_deltas": comp_deltas,
    }


# ── plots ─────────────────────────────────────────────────────────────


def plot_reward_curve(
    train_log_path: Path,
    eval_points: dict[str, float],
    out_path: Path,
) -> None:
    """training reward over steps + horizontal eval points overlaid."""
    rows = load_jsonl(train_log_path)
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        s = r.get("step")
        if s is not None and "reward" in r and "train_runtime" not in r:
            by_step[int(s)] = r
    steps = sorted(by_step)
    rewards = [by_step[s]["reward"] for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, rewards, color="#1f77b4", linewidth=1.2, label="training reward (per 10 steps)")
    for label, y in eval_points.items():
        color = {"baseline": "#888888", "ckpt100": "#ff7f0e",
                 "ckpt250": "#2ca02c", "ckpt500": "#d62728"}.get(label, "black")
        ax.axhline(y, linestyle="--", color=color, alpha=0.7, linewidth=1.2,
                   label=f"{label} held-out test mean = {y:.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("GRPO training trajectory + held-out evaluation points")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_before_after_distribution(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    out_path: Path,
    after_label: str = "AFTER (checkpoint-500)",
) -> None:
    common = sorted(set(before) & set(after))
    b = [before[k]["reward_mean"] for k in common]
    a = [after[k]["reward_mean"] for k in common]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(b, bins=bins, alpha=0.55, color="#888888", label=f"BEFORE (Qwen base)  μ={np.mean(b):.3f}")
    ax.hist(a, bins=bins, alpha=0.55, color="#d62728", label=f"{after_label}  μ={np.mean(a):.3f}")
    ax.set_xlabel("per-instance mean reward")
    ax.set_ylabel("count of seeds")
    ax.set_title("Per-seed reward distribution (paired by instance_hash)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_component_deltas(comparison: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Bar chart of component deltas per checkpoint."""
    components = ["parse_valid", "format_valid", "nmse", "support", "conformal"]
    checkpoints = [c for c in ["ckpt100", "ckpt250", "ckpt500"] if c in comparison]
    if not checkpoints:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(components))
    width = 0.25
    colors = {"ckpt100": "#ff7f0e", "ckpt250": "#2ca02c", "ckpt500": "#d62728"}
    for i, ck in enumerate(checkpoints):
        deltas = [comparison[ck]["component_deltas"].get(c, 0.0) for c in components]
        offset = (i - (len(checkpoints) - 1) / 2.0) * width
        ax.bar(x + offset, deltas, width, label=ck, color=colors.get(ck, "gray"))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("Δ component value (AFTER − BEFORE)")
    ax.set_title("Component-wise paired Δ (AFTER GRPO − BEFORE Qwen base)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory(comparison: dict[str, dict[str, Any]], baseline_mean: float,
                    baseline_ci: tuple[float, float], out_path: Path) -> None:
    """mean reward across BEFORE / ckpt100 / ckpt250 / ckpt500 with CI bars."""
    labels = ["BEFORE (M5)"]
    means = [baseline_mean]
    los = [baseline_mean - baseline_ci[0]]
    his = [baseline_ci[1] - baseline_mean]

    for ck in ["ckpt100", "ckpt250", "ckpt500"]:
        if ck not in comparison:
            continue
        labels.append(ck)
        m = comparison[ck]["mean_reward_after"]
        means.append(m)
        d_lo, d_hi = comparison[ck]["delta_ci95"]
        # Confidence band on AFTER mean ≈ baseline_mean + Δ ± half-width(Δ)
        # Half-width comes from Δ CI relative to Δ mean
        d_mean = comparison[ck]["delta_mean"]
        half_lo = d_mean - d_lo
        half_hi = d_hi - d_mean
        los.append(half_lo)
        his.append(half_hi)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, means, yerr=[los, his], fmt="o-", capsize=5, color="#1f77b4",
                ecolor="gray", linewidth=1.6, markersize=8)
    for xi, mi in zip(x, means):
        ax.text(xi, mi + 0.012, f"{mi:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean reward on TEST seeds 2000–2099")
    ax.set_title("Held-out evaluation trajectory (paired Δ from BEFORE)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--ckpt100", type=Path,
                    default=DEFAULT_OUTDIR / "qwen15b_grpo_eval_ckpt100.jsonl")
    ap.add_argument("--ckpt250", type=Path,
                    default=DEFAULT_OUTDIR / "qwen15b_grpo_eval_ckpt250.jsonl")
    ap.add_argument("--ckpt500", type=Path,
                    default=DEFAULT_OUTDIR / "qwen15b_grpo_eval_ckpt500.jsonl")
    ap.add_argument("--baseline-stats", type=Path,
                    default=DEFAULT_OUTDIR / "qwen15b_base_eval_stats.json")
    ap.add_argument("--train-log", type=Path, default=DEFAULT_TRAIN_LOG)
    ap.add_argument("--out-stats", type=Path,
                    default=DEFAULT_OUTDIR / "comparison_stats.json")
    ap.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = ap.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.baseline}...")
    baseline_rows = load_jsonl(args.baseline)
    baseline_per = per_seed_means(baseline_rows)
    baseline_means = np.array([v["reward_mean"] for v in baseline_per.values()])
    baseline_mean = float(np.mean(baseline_means))
    baseline_ci = bootstrap_ci(baseline_means)
    print(f"  {len(baseline_rows)} rows, {len(baseline_per)} unique instances; "
          f"mean_reward={baseline_mean:.4f} CI95=[{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]")

    comparison: dict[str, Any] = {}
    ckpt_paths = {"ckpt100": args.ckpt100, "ckpt250": args.ckpt250, "ckpt500": args.ckpt500}
    for label, path in ckpt_paths.items():
        if not path.exists():
            print(f"SKIP {label}: {path} missing")
            continue
        print(f"Loading {path}...")
        rows = load_jsonl(path)
        per = per_seed_means(rows)
        cmp_ = paired_compare(baseline_per, per)
        cmp_["n_eval_rows"] = len(rows)
        cmp_["n_eval_instances"] = len(per)
        comparison[label] = cmp_
        print(f"  {label}: n_overlap={cmp_['n_overlap']} "
              f"mean_after={cmp_['mean_reward_after']:.4f} "
              f"Δ={cmp_['delta_mean']:+.4f} "
              f"CI95=[{cmp_['delta_ci95'][0]:+.4f}, {cmp_['delta_ci95'][1]:+.4f}] "
              f"wilcoxon_p={cmp_['wilcoxon_pvalue']:.2e} d={cmp_['cohen_d']:.3f}")

    # Save stats.
    out = {
        "baseline": {
            "mean_reward": baseline_mean,
            "ci95": list(baseline_ci),
            "n_instances": len(baseline_per),
            "n_rows": len(baseline_rows),
        },
        "checkpoints": comparison,
    }
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out_stats}")

    # Plots.
    eval_points = {"baseline": baseline_mean}
    for ck in ["ckpt100", "ckpt250", "ckpt500"]:
        if ck in comparison:
            eval_points[ck] = comparison[ck]["mean_reward_after"]

    plot_reward_curve(args.train_log, eval_points, args.plots_dir / "reward_curve.png")
    print(f"Wrote {args.plots_dir / 'reward_curve.png'}")

    if "ckpt500" in comparison:
        ck500_per = per_seed_means(load_jsonl(ckpt_paths["ckpt500"]))
        plot_before_after_distribution(
            baseline_per, ck500_per,
            args.plots_dir / "before_after_distribution.png",
        )
        print(f"Wrote {args.plots_dir / 'before_after_distribution.png'}")

    plot_component_deltas(comparison, args.plots_dir / "component_deltas.png")
    print(f"Wrote {args.plots_dir / 'component_deltas.png'}")

    plot_trajectory(comparison, baseline_mean, baseline_ci,
                    args.plots_dir / "trajectory.png")
    print(f"Wrote {args.plots_dir / 'trajectory.png'}")

    # Stdout summary table.
    print()
    print("=== SUMMARY TABLE ===")
    hdr = f"{'run':<14} {'n':>4} {'mean_reward':>12} {'Δ vs BEFORE':>13} {'CI95':>22} {'p':>10} {'d':>6}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'BEFORE (M5)':<14} {len(baseline_per):>4} {baseline_mean:>12.4f} "
          f"{'—':>13} {f'[{baseline_ci[0]:.3f}, {baseline_ci[1]:.3f}]':>22} {'—':>10} {'—':>6}")
    for ck, cmp_ in comparison.items():
        ci = cmp_["delta_ci95"]
        print(f"{ck:<14} {cmp_['n_overlap']:>4} {cmp_['mean_reward_after']:>12.4f} "
              f"{cmp_['delta_mean']:>+13.4f} {f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22} "
              f"{cmp_['wilcoxon_pvalue']:>10.2e} {cmp_['cohen_d']:>+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
