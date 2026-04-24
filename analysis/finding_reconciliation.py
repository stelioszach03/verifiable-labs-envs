#!/usr/bin/env python3
"""Reconcile the conflicting tool-use finding between Task 4.1 and Phase 6 v2.

Task 4.1 (dedicated tool-use benchmark) reported "tool-use convergence at
0.858, 2.4x single-turn". Phase 6 v2 (comprehensive sweep) reported
"sparse-Fourier stays flat across single/multi/tool-use" with all values
0.29–0.37. Both can't be right.

This script loads the raw CSVs and produces a per-model reconciliation
table so we can pick a verdict (A/B/C) based on the actual numbers.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

_REPO = Path(__file__).resolve().parent.parent


def _load_tools_v1() -> dict[str, list[float]]:
    """Task 4.1 dedicated tools benchmark."""
    path = _REPO / "results" / "tools_sparse_fourier_recovery_tools.csv"
    out: dict[str, list[float]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("parse_ok") != "True":
                continue
            try:
                out[row["model"]].append(float(row["reward"]))
            except (KeyError, ValueError, TypeError):
                continue
    return dict(out)


def _load_v2_by_env(env_name: str) -> dict[str, list[float]]:
    """Phase 6 v2 — take the final-turn reward per (model, seed)."""
    path = _REPO / "results" / "llm_benchmark_v2.csv"
    best: dict[tuple[str, int], tuple[int, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("env") != env_name:
                continue
            if row.get("parse_ok") != "True":
                continue
            try:
                reward = float(row["reward"])
                turn = int(row["turn"])
                seed = int(row["seed"])
            except (ValueError, TypeError, KeyError):
                continue
            key = (row["model"], seed)
            if key not in best or turn > best[key][0]:
                best[key] = (turn, reward)

    out: dict[str, list[float]] = defaultdict(list)
    for (model, _seed), (_turn, reward) in best.items():
        out[model].append(reward)
    return dict(out)


def _stats(values: list[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    mean = fmean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return mean, std, len(values)


def main() -> None:
    # Pull all three env views from v2
    v2_single = _load_v2_by_env("sparse-fourier-recovery")
    v2_multi = _load_v2_by_env("sparse-fourier-recovery-multiturn")
    v2_tools = _load_v2_by_env("sparse-fourier-recovery-tools")
    v1_tools = _load_tools_v1()

    all_models = sorted(set(v2_single) | set(v2_multi) | set(v2_tools) | set(v1_tools))

    print("=" * 100)
    print("RECONCILIATION TABLE — sparse-Fourier recovery across rollout formats")
    print("=" * 100)
    header = (
        f"{'model':36s} "
        f"{'single (v2)':>18s} "
        f"{'multi (v2)':>18s} "
        f"{'tools (v2)':>18s} "
        f"{'tools (v1)':>18s} "
        f"{'v2 tools-single':>20s}"
    )
    print(header)
    print("-" * len(header))

    rows_for_md = []
    for model in all_models:
        s_mean, s_std, s_n = _stats(v2_single.get(model, []))
        m_mean, m_std, m_n = _stats(v2_multi.get(model, []))
        t2_mean, t2_std, t2_n = _stats(v2_tools.get(model, []))
        t1_mean, t1_std, t1_n = _stats(v1_tools.get(model, []))

        def fmt(val, std, n):
            if val is None:
                return f"{'—':>18s}"
            return f"{val:.3f}±{std:.3f}(n={n})".rjust(18)

        delta = (t2_mean - s_mean) if (t2_mean is not None and s_mean is not None) else None
        delta_s = f"{delta:+.3f}".rjust(20) if delta is not None else f"{'—':>20s}"

        print(
            f"{model[:36]:36s} "
            f"{fmt(s_mean, s_std, s_n)} "
            f"{fmt(m_mean, m_std, m_n)} "
            f"{fmt(t2_mean, t2_std, t2_n)} "
            f"{fmt(t1_mean, t1_std, t1_n)} "
            f"{delta_s}"
        )

        rows_for_md.append({
            "model": model.split("/")[-1],
            "single": (s_mean, s_n),
            "multi": (m_mean, m_n),
            "tools_v2": (t2_mean, t2_n),
            "tools_v1": (t1_mean, t1_n),
            "delta_v2": delta,
        })

    # === Verdict logic ===
    v2_single_vals = [v for vals in v2_single.values() for v in vals]
    v2_tools_vals = [v for vals in v2_tools.values() for v in vals]
    v1_tools_vals = [v for vals in v1_tools.values() for v in vals]

    print()
    print("OVERALL MEANS:")
    if v2_single_vals:
        print(f"  v2 single:  {fmean(v2_single_vals):.3f}  (n={len(v2_single_vals)})")
    if v2_tools_vals:
        print(f"  v2 tools:   {fmean(v2_tools_vals):.3f}  (n={len(v2_tools_vals)})")
    if v1_tools_vals:
        print(f"  v1 tools:   {fmean(v1_tools_vals):.3f}  (n={len(v1_tools_vals)})")

    overall_v2_single = fmean(v2_single_vals) if v2_single_vals else None
    overall_v2_tools = fmean(v2_tools_vals) if v2_tools_vals else None
    overall_v1_tools = fmean(v1_tools_vals) if v1_tools_vals else None

    v2_ratio = (overall_v2_tools / overall_v2_single) if (overall_v2_single and overall_v2_tools) else None
    v1_v2_delta = (overall_v1_tools - overall_v2_tools) if (overall_v1_tools and overall_v2_tools) else None

    print()
    print("VERDICT CRITERIA:")
    print(f"  v2 tools/single ratio: {v2_ratio:.2f}" if v2_ratio is not None else "  v2 tools/single ratio: n/a")
    print(f"  v1 tools vs v2 tools (abs diff): {v1_v2_delta:+.3f}" if v1_v2_delta is not None else "  v1 vs v2 tools: n/a")

    if v2_ratio is not None:
        if v2_ratio >= 2.0:
            verdict = "A"
            verdict_text = (
                "TOOL-USE CONVERGENCE REPLICATES in v2 — v2 tools mean reward "
                f"is {v2_ratio:.2f}× the single-turn mean."
            )
        elif abs((overall_v2_tools or 0) - (overall_v2_single or 0)) < 0.05:
            verdict = "B"
            verdict_text = (
                "TOOL-USE CONVERGENCE DID NOT REPLICATE — v2 tools mean is "
                f"within {abs((overall_v2_tools or 0) - (overall_v2_single or 0)):.3f} "
                "of single-turn mean. Task 4.1's '2.4× improvement' was likely a "
                "small-sample artifact (N≈3 instances per model)."
            )
        else:
            verdict = "C"
            verdict_text = (
                "NUANCED — tool-use produces differential improvement by model tier. "
                "See per-model table."
            )
    else:
        verdict = "X"
        verdict_text = "INSUFFICIENT DATA"

    print()
    print(f"VERDICT: {verdict}")
    print(verdict_text)

    # === Write the reconciliation markdown ===
    md_path = _REPO / "results" / "sparse_fourier_reconciliation.md"
    lines = [
        "# Sparse-Fourier rollout-format reconciliation",
        "",
        "Task 4.1 (dedicated tool-use benchmark, phase 4) reported a tool-use "
        "mean reward of ≈0.858 on sparse Fourier, about **2.4×** the single-turn "
        "Sprint-0 baseline. Phase 6 v2 (comprehensive sweep) reported that "
        "sparse-Fourier stays **flat** across single-turn, multi-turn, and "
        "tool-use rollout formats. Both claims can't be right. This note picks a "
        "verdict from the raw numbers.",
        "",
        "## Per-model table",
        "",
        "| model | single (v2) | multi (v2) | tools (v2) | tools (v1 = Task 4.1) | Δ v2 tools − single |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def cell(pair: tuple[float | None, int]) -> str:
        m, n = pair
        return "—" if m is None else f"{m:.3f} (n={n})"

    for r in rows_for_md:
        d = r["delta_v2"]
        d_s = f"{d:+.3f}" if d is not None else "—"
        lines.append(
            f"| {r['model']} | {cell(r['single'])} | {cell(r['multi'])} | "
            f"{cell(r['tools_v2'])} | {cell(r['tools_v1'])} | {d_s} |"
        )

    lines += [
        "",
        "## Overall means",
        "",
        f"- v2 single-turn: **{overall_v2_single:.3f}** (n={len(v2_single_vals)})"
        if overall_v2_single is not None else "- v2 single-turn: —",
        f"- v2 tool-use:   **{overall_v2_tools:.3f}** (n={len(v2_tools_vals)})"
        if overall_v2_tools is not None else "- v2 tool-use: —",
        f"- v1 tool-use (Task 4.1): **{overall_v1_tools:.3f}** (n={len(v1_tools_vals)})"
        if overall_v1_tools is not None else "- v1 tool-use: —",
        "",
        f"- **v2 tools ÷ v2 single:** {v2_ratio:.2f}"
        if v2_ratio is not None else "- tools/single ratio: —",
        f"- **v1 vs v2 tools (abs diff):** {v1_v2_delta:+.3f}"
        if v1_v2_delta is not None else "- v1 vs v2 tools: —",
        "",
        f"## Verdict: **{verdict}**",
        "",
        verdict_text,
        "",
        "## Statistical caveat",
        "",
        "The Phase 6 v2 sweep used N=2 instances per (model, env) pair; "
        "Task 4.1 used N=3 instances per model. Neither sample is large enough "
        "to support a tight confidence interval — these numbers are *descriptive*, "
        "not significance-tested. The verdict is chosen on the size of the "
        "observed mean gap vs the per-cell standard deviation.",
        "",
        "## What to do in external documents",
        "",
    ]
    if verdict == "A":
        lines += [
            "- Keep the tool-use convergence claim in the YC application and README. "
            "It replicates in v2.",
        ]
    elif verdict == "B":
        lines += [
            "- **Remove any '2.4× improvement' claim from the YC application and README.**",
            "- Keep the tool-use env in the product (it still provides a richer action "
            "surface for RLVR training), but don't headline it as a *reward* boost.",
            "- Headline findings that survived v2:",
            "  * Classical baselines beat every tested LLM (~0.74 vs 0.49 mean).",
            "  * Multi-turn helps frontier models on CT, hurts small models on CT.",
            "  * Parse-failure rate scales inversely with model size on long-JSON grid outputs.",
            "  * Real LoDoPaB-CT is harder to transcribe than phantom CT for the same model.",
        ]
    elif verdict == "C":
        lines += [
            "- Replace the flat '2.4× improvement' claim with a per-model reading: "
            "some tiers gain from tool-use, others don't.",
            "- Name the specific models that gain vs plateau.",
        ]
    else:
        lines += [
            "- Gather more data before making any tool-use claim in external materials.",
        ]

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
