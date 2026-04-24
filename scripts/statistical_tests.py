#!/usr/bin/env python3
"""Paired-bootstrap significance tests: classical vs each LLM, per env.

For each (env, model) pair with LLM benchmark rows in results/:
- Compute per-seed LLM reward.
- Compute per-seed classical-baseline reward on the SAME seeds
  (re-running the env's run_baseline on each seed).
- Paired-bootstrap (10_000 resamples) of Δ = classical − LLM.
- Report mean Δ, 95 % CI, and two-sided bootstrap p-value
  (fraction of resamples with sign flipped vs observed).

Output: results/stat_tests.csv with one row per (env, model) comparison.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifiable_labs_envs import load_environment  # noqa: E402


# Source CSVs with LLM rewards (final-turn reward per (env, model, seed)).
CSVS = [
    REPO / "results" / "llm_benchmark_v2.csv",       # Sprint-1 v2 sweep
    REPO / "results" / "meta_benchmark_v3.csv",       # sprint-giga meta
    REPO / "results" / "phase_retrieval_v1_benchmark.csv",
    REPO / "results" / "mri_knee_v1_benchmark.csv",
    REPO / "results" / "opus_nano_fill_v2.csv",       # sprint-paper fill-in
]


def _llm_final_turn_rewards() -> dict[tuple[str, str], dict[int, float]]:
    """Merge all LLM CSVs into (env, model) → {seed: final-turn reward}.

    v2 schema has ``seed`` + ``turn`` columns; other CSVs have
    ``instance_id`` (= seed), and for multiturn we take the final turn
    implicitly (row already represents the final answer). We take the
    max-turn row per (env, model, seed) when ``turn`` is present.
    """
    best: dict[tuple[str, str, int], tuple[int, float]] = {}
    for csv_path in CSVS:
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("parse_ok") != "True":
                    continue
                try:
                    reward = float(r["reward"])
                except (TypeError, ValueError):
                    continue
                # Schema dispatch: v2 uses "seed"+"turn"; giga CSVs use "instance_id".
                seed_raw = r.get("seed") or r.get("instance_id")
                try:
                    seed = int(seed_raw)
                except (TypeError, ValueError):
                    continue
                turn = int(r.get("turn") or 1)
                key = (r["env"], r["model"], seed)
                if key not in best or turn > best[key][0]:
                    best[key] = (turn, reward)

    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (env, model, seed), (_, reward) in best.items():
        out[(env, model)][seed] = reward
    return out


def _classical_rewards_for(env_id: str, seeds: list[int]) -> dict[int, float]:
    """Run the env's classical baseline on each seed, return (seed → reward)."""
    try:
        env = load_environment(env_id)
    except Exception:
        return {}
    out: dict[int, float] = {}
    for seed in seeds:
        try:
            r = env.run_baseline(seed=seed)
            out[seed] = float(r["reward"])
        except Exception:
            continue
    return out


def _paired_bootstrap(deltas: np.ndarray, n_resamples: int = 10_000,
                      seed: int = 42) -> tuple[float, float, float, float]:
    """Return (mean_delta, ci_low, ci_high, two_sided_p).

    p = 2 * min(P(mean >= 0), P(mean < 0)) under the null of no difference.
    """
    rng = np.random.default_rng(seed)
    n = deltas.size
    resamples = rng.choice(deltas, size=(n_resamples, n), replace=True)
    boot_means = resamples.mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    observed = float(deltas.mean())
    # Two-sided bootstrap p via re-centring the distribution at 0.
    centred = boot_means - observed
    p_two_sided = 2.0 * float(np.minimum(
        np.mean(centred >= abs(observed)),
        np.mean(centred <= -abs(observed)),
    ))
    return observed, float(ci_low), float(ci_high), min(p_two_sided, 1.0)


def main() -> None:
    llm_rewards = _llm_final_turn_rewards()
    rows = []
    # Limit to envs that actually have a run_baseline (i.e. all 10 shipped).
    for (env_id, model), llm_by_seed in sorted(llm_rewards.items()):
        classical = _classical_rewards_for(env_id, sorted(llm_by_seed))
        # Pair seeds present in both
        paired = [
            (llm_by_seed[s], classical[s])
            for s in llm_by_seed
            if s in classical
        ]
        if len(paired) < 2:  # nothing to bootstrap
            continue
        llm_arr = np.asarray([p[0] for p in paired])
        cls_arr = np.asarray([p[1] for p in paired])
        deltas = cls_arr - llm_arr  # positive means classical wins
        mean_d, lo, hi, p = _paired_bootstrap(deltas)
        rows.append({
            "env": env_id,
            "model": model,
            "n_paired": len(paired),
            "mean_classical": round(float(cls_arr.mean()), 4),
            "mean_llm": round(float(llm_arr.mean()), 4),
            "mean_delta": round(mean_d, 4),
            "ci_low_95": round(lo, 4),
            "ci_high_95": round(hi, 4),
            "p_two_sided": round(p, 4),
        })
        print(
            f"{env_id:38s} {model:30s} "
            f"n={len(paired)} Δ={mean_d:+.3f} "
            f"95%CI=[{lo:+.3f},{hi:+.3f}] p={p:.3g}"
        )

    out = REPO / "results" / "stat_tests.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "env", "model", "n_paired", "mean_classical", "mean_llm",
            "mean_delta", "ci_low_95", "ci_high_95", "p_two_sided",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} comparisons)")

    # Aggregate: what fraction of comparisons have classical > LLM at p<0.05?
    if rows:
        total = len(rows)
        wins = sum(
            1 for r in rows
            if r["mean_delta"] > 0 and r["p_two_sided"] < 0.05
        )
        ties = sum(1 for r in rows if r["p_two_sided"] >= 0.05)
        losses = sum(
            1 for r in rows
            if r["mean_delta"] < 0 and r["p_two_sided"] < 0.05
        )
        mean_delta = fmean(r["mean_delta"] for r in rows)
        print(f"\nclassical significantly better (p<0.05): {wins}/{total}")
        print(f"no significant difference:               {ties}/{total}")
        print(f"LLM significantly better:                 {losses}/{total}")
        print(f"mean Δ (classical − LLM) across all:      {mean_delta:+.3f}")


if __name__ == "__main__":
    main()
