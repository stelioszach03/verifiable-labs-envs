"""Regression-detection module for monitor runs (Phase 28.D).

PHASE_28_PLAN.md §10 — D5-C primary (conformal drift) + D5-A
secondary (paired-sample bootstrap) with the combined-verdict
matrix:

    | Conformal verdict | Bootstrap verdict | Combined        |
    | regressed         | (any)             | regressed       |
    | warning           | regressed=True    | regressed       |
    | warning           | otherwise         | warning         |
    | ok                | regressed=True    | warning         |
    | ok                | otherwise         | ok              |

The conformal verdict targets the env-level conformal coverage gap
(absolute, against the locked target ``1 - alpha = 0.90``); the
bootstrap verdict tests for a statistically significant negative
mean-reward delta against the baseline run's per-episode rewards.
"""
from __future__ import annotations

from typing import Any, Literal

ConformalVerdict = Literal["ok", "warning", "regressed"]
CombinedVerdict = Literal["ok", "warning", "regressed"]

# D5-C tolerance: 5pp absolute coverage gap = warning, 10pp = regressed.
DEFAULT_TOLERANCE = 0.05
DEFAULT_REGRESSED_TOLERANCE = 0.10
DEFAULT_TARGET_COVERAGE = 0.90
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_P_THRESHOLD = 0.05


# ── conformal verdict (D5-C) ───────────────────────────────────────


def conformal_drift_verdict(
    *,
    current_coverage: float | None,
    baseline_coverage: float | None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    tolerance: float = DEFAULT_TOLERANCE,
    regressed_tolerance: float = DEFAULT_REGRESSED_TOLERANCE,
) -> dict[str, Any]:
    """Return ``{"verdict", "current", "baseline", "delta_to_target", ...}``.

    ``current_coverage = None`` (no episodes recorded) returns
    ``"failed"``-equivalent — the calling worker treats that as an
    ``error_excerpt`` rather than a regression alert.
    """
    if current_coverage is None:
        return {
            "verdict": "warning",
            "current": None,
            "baseline": baseline_coverage,
            "delta_to_target": None,
            "delta_to_baseline": None,
        }

    delta_target = current_coverage - target_coverage
    if abs(delta_target) > regressed_tolerance:
        verdict: ConformalVerdict = "regressed"
    elif baseline_coverage is None:
        # First run — no baseline yet. Apply target-only tolerance.
        verdict = "ok" if abs(delta_target) <= tolerance else "warning"
    elif abs(current_coverage - baseline_coverage) > tolerance:
        verdict = "warning"
    else:
        verdict = "ok"
    return {
        "verdict": verdict,
        "current": float(current_coverage),
        "baseline": (
            float(baseline_coverage) if baseline_coverage is not None else None
        ),
        "delta_to_target": float(delta_target),
        "delta_to_baseline": (
            float(current_coverage - baseline_coverage)
            if baseline_coverage is not None
            else None
        ),
    }


# ── bootstrap verdict (D5-A) ───────────────────────────────────────


def _resample_indices(rng_seed: int, n: int, n_resamples: int) -> list[list[int]]:
    """Deterministic bootstrap indices via a stdlib RNG.

    Avoids a numpy hard-dep in services/api. Random.Random is
    seedable and produces identical sequences across Python versions.
    """
    import random

    rng = random.Random(int(rng_seed) & 0xFFFFFFFF)
    out: list[list[int]] = []
    for _ in range(n_resamples):
        out.append([rng.randint(0, n - 1) for _ in range(n)])
    return out


def bootstrap_reward_delta(
    *,
    current_rewards: list[float],
    baseline_rewards: list[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = 0,
    p_threshold: float = DEFAULT_P_THRESHOLD,
) -> dict[str, Any]:
    """Paired-sample bootstrap on per-episode reward differences.

    Truncates to the shorter of the two vectors so the pairing is
    well-defined even when the current run shipped a different
    episode count from the baseline.

    Returns ``{"mean_delta", "ci_low", "ci_high", "p_value", "regressed"}``.
    A negative mean delta + ``p < p_threshold`` ⇒ regression detected.
    """
    n = min(len(current_rewards), len(baseline_rewards))
    if n == 0:
        return {
            "mean_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "regressed": False,
        }
    paired = [
        float(current_rewards[i]) - float(baseline_rewards[i])
        for i in range(n)
    ]
    mean_delta = sum(paired) / n

    # Degenerate identical-arrays case — bootstrap distribution is
    # all zeros, p=1.0, regressed=False.
    if all(d == 0.0 for d in paired):
        return {
            "mean_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "regressed": False,
        }

    deltas: list[float] = []
    for indices in _resample_indices(rng_seed, n, n_resamples):
        s = 0.0
        for idx in indices:
            s += paired[idx]
        deltas.append(s / n)
    deltas.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    ci_low = float(deltas[lo_idx])
    ci_high = float(deltas[hi_idx])
    # Two-sided p-value approximation via the percentile method:
    # fraction of resamples on the wrong side of zero relative to the
    # observed mean.
    if mean_delta >= 0:
        p_value = 2.0 * (sum(1 for d in deltas if d <= 0) / n_resamples)
    else:
        p_value = 2.0 * (sum(1 for d in deltas if d >= 0) / n_resamples)
    p_value = min(1.0, p_value)
    regressed = bool(p_value < p_threshold and ci_high < 0.0)
    return {
        "mean_delta": float(mean_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "regressed": regressed,
    }


# ── combined verdict ───────────────────────────────────────────────


def _combine(
    conformal_verdict: ConformalVerdict,
    bootstrap_regressed: bool,
) -> CombinedVerdict:
    if conformal_verdict == "regressed":
        return "regressed"
    if conformal_verdict == "warning" and bootstrap_regressed:
        return "regressed"
    if conformal_verdict == "warning":
        return "warning"
    if bootstrap_regressed:  # conformal == "ok"
        return "warning"
    return "ok"


def _flatten_rewards(summary: dict[str, Any]) -> list[float]:
    """Concatenate per-env reward vectors into a single flat list."""
    out: list[float] = []
    per_env = summary.get("per_env") or {}
    for env_id in sorted(per_env):
        out.extend(float(r) for r in per_env[env_id].get("rewards") or [])
    return out


def compute_verdict(
    *,
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Combine D5-C + D5-A into the canonical verdict payload.

    Shape::

        {
          "verdict":  "ok" | "warning" | "regressed",
          "conformal": {...},
          "bootstrap": {...},
          "per_env_breakdown": [{"env_id", "current_mean", "baseline_mean",
                                  "delta", "regressed"}, ...],
          "rng_seed": int,
        }
    """
    current_cov = current_summary.get("overall_coverage")
    baseline_cov = (
        baseline_summary.get("overall_coverage")
        if baseline_summary
        else None
    )
    conformal = conformal_drift_verdict(
        current_coverage=current_cov,
        baseline_coverage=baseline_cov,
    )

    if baseline_summary is None:
        bootstrap = {
            "mean_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "regressed": False,
        }
    else:
        cur_rewards = _flatten_rewards(current_summary)
        base_rewards = _flatten_rewards(baseline_summary)
        bootstrap = bootstrap_reward_delta(
            current_rewards=cur_rewards,
            baseline_rewards=base_rewards,
            rng_seed=rng_seed,
        )

    combined = _combine(
        conformal["verdict"], bool(bootstrap["regressed"])
    )

    per_env_breakdown: list[dict[str, Any]] = []
    cur_per_env = current_summary.get("per_env") or {}
    base_per_env = (
        (baseline_summary or {}).get("per_env") or {}
    )
    for env_id in sorted(cur_per_env):
        cur_stats = cur_per_env[env_id]
        cur_mean = cur_stats.get("mean_reward", 0.0)
        if env_id in base_per_env:
            base_mean = base_per_env[env_id].get("mean_reward", 0.0)
            cur_rewards = cur_stats.get("rewards") or []
            base_rewards = base_per_env[env_id].get("rewards") or []
            env_boot = bootstrap_reward_delta(
                current_rewards=[float(r) for r in cur_rewards],
                baseline_rewards=[float(r) for r in base_rewards],
                rng_seed=rng_seed,
                n_resamples=2_000,  # cheaper for per-env
            )
            per_env_breakdown.append(
                {
                    "env_id": env_id,
                    "current_mean": float(cur_mean),
                    "baseline_mean": float(base_mean),
                    "delta": float(cur_mean - base_mean),
                    "regressed": bool(env_boot["regressed"]),
                }
            )
        else:
            per_env_breakdown.append(
                {
                    "env_id": env_id,
                    "current_mean": float(cur_mean),
                    "baseline_mean": None,
                    "delta": None,
                    "regressed": False,
                }
            )

    return {
        "verdict": combined,
        "conformal": conformal,
        "bootstrap": bootstrap,
        "per_env_breakdown": per_env_breakdown,
        "rng_seed": int(rng_seed),
    }


__all__ = [
    "DEFAULT_TARGET_COVERAGE",
    "DEFAULT_TOLERANCE",
    "DEFAULT_REGRESSED_TOLERANCE",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_P_THRESHOLD",
    "ConformalVerdict",
    "CombinedVerdict",
    "conformal_drift_verdict",
    "bootstrap_reward_delta",
    "compute_verdict",
]
