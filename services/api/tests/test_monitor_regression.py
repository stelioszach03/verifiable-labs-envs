"""Tests for the Phase 28.D regression-detection module."""
from __future__ import annotations

import pytest

from vlabs_api.monitor_regression import (
    DEFAULT_REGRESSED_TOLERANCE,
    DEFAULT_TARGET_COVERAGE,
    DEFAULT_TOLERANCE,
    bootstrap_reward_delta,
    compute_verdict,
    conformal_drift_verdict,
)

# ── conformal verdict (D5-C) ───────────────────────────────────────


def test_conformal_verdict_target_match_returns_ok() -> None:
    out = conformal_drift_verdict(
        current_coverage=0.90, baseline_coverage=0.90,
    )
    assert out["verdict"] == "ok"


def test_conformal_verdict_just_below_tolerance_returns_warning() -> None:
    """5pp gap from baseline → warning (matches DEFAULT_TOLERANCE)."""
    out = conformal_drift_verdict(
        current_coverage=0.84, baseline_coverage=0.90,
    )
    assert out["verdict"] == "warning"


def test_conformal_verdict_above_regressed_tolerance_returns_regressed() -> None:
    """11pp gap from target (1-alpha=0.90) → regressed."""
    out = conformal_drift_verdict(
        current_coverage=0.79, baseline_coverage=0.90,
    )
    assert out["verdict"] == "regressed"


def test_conformal_verdict_no_baseline_uses_target_only() -> None:
    """First run (no baseline) compares to target=0.90 only."""
    ok = conformal_drift_verdict(current_coverage=0.91, baseline_coverage=None)
    assert ok["verdict"] == "ok"
    warn = conformal_drift_verdict(
        current_coverage=0.84, baseline_coverage=None,
    )
    assert warn["verdict"] == "warning"


def test_conformal_verdict_none_current_coverage_returns_warning() -> None:
    out = conformal_drift_verdict(
        current_coverage=None, baseline_coverage=0.90,
    )
    assert out["verdict"] == "warning"


def test_conformal_verdict_payload_carries_deltas() -> None:
    out = conformal_drift_verdict(
        current_coverage=0.85, baseline_coverage=0.90,
    )
    assert "delta_to_target" in out
    assert out["delta_to_target"] == pytest.approx(0.85 - 0.90)
    assert out["delta_to_baseline"] == pytest.approx(0.85 - 0.90)


def test_conformal_constants_match_phase_plan() -> None:
    assert pytest.approx(0.90) == DEFAULT_TARGET_COVERAGE
    assert pytest.approx(0.05) == DEFAULT_TOLERANCE
    assert pytest.approx(0.10) == DEFAULT_REGRESSED_TOLERANCE


# ── bootstrap (D5-A) ───────────────────────────────────────────────


def test_bootstrap_identical_arrays_returns_zero_delta() -> None:
    cur = [0.5, 0.6, 0.7]
    out = bootstrap_reward_delta(
        current_rewards=cur, baseline_rewards=cur, n_resamples=200, rng_seed=0,
    )
    assert out["mean_delta"] == 0.0
    assert out["regressed"] is False
    assert out["p_value"] == 1.0


def test_bootstrap_clear_regression_flagged() -> None:
    """A 0.5-pp drop with n=20 episodes is statistically significant."""
    baseline = [0.9] * 20
    current = [0.4] * 20
    out = bootstrap_reward_delta(
        current_rewards=current, baseline_rewards=baseline,
        n_resamples=2_000, rng_seed=42,
    )
    assert out["mean_delta"] == pytest.approx(-0.5)
    assert out["ci_high"] < 0
    assert out["regressed"] is True


def test_bootstrap_small_drop_not_flagged() -> None:
    """A small noisy drop (n=5) with high variance is NOT regressed."""
    baseline = [0.5, 0.7, 0.4, 0.6, 0.5]
    current = [0.45, 0.65, 0.40, 0.55, 0.50]
    out = bootstrap_reward_delta(
        current_rewards=current, baseline_rewards=baseline,
        n_resamples=2_000, rng_seed=0,
    )
    # Mean delta is small; CI may straddle zero — should NOT regress.
    assert out["mean_delta"] < 0
    # We don't assert on p_value (depends on noise), only on
    # the 'regressed' boolean for clear signal.
    if out["ci_high"] >= 0:
        assert out["regressed"] is False


def test_bootstrap_empty_arrays_returns_zero() -> None:
    out = bootstrap_reward_delta(
        current_rewards=[], baseline_rewards=[],
        n_resamples=100, rng_seed=0,
    )
    assert out["mean_delta"] == 0.0
    assert out["regressed"] is False


def test_bootstrap_rng_seed_determinism() -> None:
    cur = [0.4, 0.5, 0.6]
    base = [0.5, 0.6, 0.7]
    a = bootstrap_reward_delta(
        current_rewards=cur, baseline_rewards=base,
        n_resamples=500, rng_seed=99,
    )
    b = bootstrap_reward_delta(
        current_rewards=cur, baseline_rewards=base,
        n_resamples=500, rng_seed=99,
    )
    assert a["mean_delta"] == b["mean_delta"]
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]


def test_bootstrap_truncates_unequal_lengths() -> None:
    """Mismatched lengths are pair-truncated to the shorter vector."""
    cur = [0.5, 0.6]
    base = [0.5, 0.6, 0.7, 0.8, 0.9]
    out = bootstrap_reward_delta(
        current_rewards=cur, baseline_rewards=base,
        n_resamples=200, rng_seed=0,
    )
    # Only the first 2 paired diffs (0.0, 0.0) are used.
    assert out["mean_delta"] == 0.0


# ── combined verdict matrix ─────────────────────────────────────────


def _summary(coverage: float, rewards: list[float],
             env_id: str = "math-algebra") -> dict:
    return {
        "per_env": {
            env_id: {
                "n": len(rewards),
                "mean_reward": (
                    sum(rewards) / len(rewards) if rewards else 0.0
                ),
                "rewards": list(rewards),
                "coverage_flags": [coverage] * len(rewards),
                "coverage": coverage,
            }
        },
        "overall_mean_reward": (
            sum(rewards) / len(rewards) if rewards else 0.0
        ),
        "overall_coverage": coverage,
        "n_total": len(rewards),
    }


def test_combined_verdict_ok_when_both_signals_clean() -> None:
    cur = _summary(0.90, [0.6, 0.6, 0.6])
    base = _summary(0.90, [0.6, 0.6, 0.6])
    out = compute_verdict(current_summary=cur, baseline_summary=base)
    assert out["verdict"] == "ok"


def test_combined_verdict_warning_when_only_conformal_drifts() -> None:
    """Coverage gap 7pp from baseline; rewards identical → warning."""
    cur = _summary(0.83, [0.6, 0.6, 0.6])
    base = _summary(0.90, [0.6, 0.6, 0.6])
    out = compute_verdict(current_summary=cur, baseline_summary=base)
    assert out["conformal"]["verdict"] == "warning"
    assert out["bootstrap"]["regressed"] is False
    assert out["verdict"] == "warning"


def test_combined_verdict_regressed_when_conformal_breaches_hard_cap() -> None:
    """Conformal beyond 10pp → regressed regardless of bootstrap."""
    cur = _summary(0.78, [0.5, 0.5, 0.5])
    base = _summary(0.90, [0.5, 0.5, 0.5])
    out = compute_verdict(current_summary=cur, baseline_summary=base)
    assert out["conformal"]["verdict"] == "regressed"
    assert out["verdict"] == "regressed"


def test_combined_verdict_regressed_when_warning_plus_bootstrap() -> None:
    """Conformal 6pp gap + bootstrap-confirmed reward drop → regressed."""
    cur = _summary(0.83, [0.4] * 20)
    base = _summary(0.90, [0.8] * 20)
    out = compute_verdict(
        current_summary=cur, baseline_summary=base, rng_seed=7,
    )
    assert out["conformal"]["verdict"] == "warning"
    assert out["bootstrap"]["regressed"] is True
    assert out["verdict"] == "regressed"


def test_combined_verdict_warning_when_only_bootstrap_flags() -> None:
    """Conformal ok + bootstrap regression → warning (not regressed)."""
    cur = _summary(0.91, [0.4] * 20)  # coverage near target — ok
    base = _summary(0.90, [0.8] * 20)
    out = compute_verdict(
        current_summary=cur, baseline_summary=base, rng_seed=11,
    )
    assert out["conformal"]["verdict"] == "ok"
    assert out["bootstrap"]["regressed"] is True
    assert out["verdict"] == "warning"


def test_combined_verdict_handles_none_baseline() -> None:
    """First run (no baseline) — bootstrap=zero, conformal target-only."""
    cur = _summary(0.91, [0.5, 0.5, 0.5])
    out = compute_verdict(current_summary=cur, baseline_summary=None)
    assert out["verdict"] in ("ok", "warning")
    assert out["bootstrap"]["regressed"] is False


def test_per_env_breakdown_shape() -> None:
    cur = _summary(0.90, [0.5, 0.5], env_id="math-algebra")
    base = _summary(0.90, [0.5, 0.5], env_id="math-algebra")
    out = compute_verdict(current_summary=cur, baseline_summary=base)
    breakdown = out["per_env_breakdown"]
    assert len(breakdown) == 1
    entry = breakdown[0]
    assert entry["env_id"] == "math-algebra"
    assert entry["current_mean"] == pytest.approx(0.5)
    assert entry["baseline_mean"] == pytest.approx(0.5)
    assert entry["regressed"] is False


def test_per_env_breakdown_handles_unmatched_envs() -> None:
    """Env in current but not baseline shows None for baseline_mean."""
    cur = _summary(0.90, [0.5, 0.5], env_id="math-algebra")
    cur["per_env"]["code-humaneval"] = {
        "n": 1,
        "mean_reward": 0.7,
        "rewards": [0.7],
        "coverage_flags": [1.0],
        "coverage": 1.0,
    }
    base = _summary(0.90, [0.5, 0.5], env_id="math-algebra")
    out = compute_verdict(current_summary=cur, baseline_summary=base)
    new_env = next(e for e in out["per_env_breakdown"] if e["env_id"] == "code-humaneval")
    assert new_env["baseline_mean"] is None
    assert new_env["delta"] is None
