"""Tests for vlabs_calibrate.calibrate() and CalibratedRewardFn."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import vlabs_calibrate as vc
from vlabs_calibrate import nonconformity as nc
from vlabs_calibrate.types import CalibrationResult, CoverageReport


def _identity_reward(*, x: float) -> float:
    return float(x)


def test_calibrate_returns_callable_dataclass(gaussian_traces) -> None:
    traces = gaussian_traces(200, seed=0, sigma=0.5)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    assert callable(cal)
    assert isinstance(cal, vc.CalibratedRewardFn)
    assert 0 < cal.quantile < 10
    assert cal.alpha == pytest.approx(0.1)
    assert cal.n_calibration == 200
    assert cal.nonconformity_name == "scaled_residual"
    assert cal.target_coverage == pytest.approx(0.9)
    assert "mean" in cal.nonconformity_stats


def test_call_returns_calibration_result(gaussian_traces) -> None:
    traces = gaussian_traces(200, seed=1, sigma=0.5)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    result = cal(x=1.0, sigma=0.5)
    assert isinstance(result, CalibrationResult)
    assert result.reward == pytest.approx(1.0)
    lo, hi = result.interval
    assert lo < result.reward < hi
    assert result.target_coverage == pytest.approx(0.9)
    assert result.covered is None


def test_call_with_reference_sets_covered_flag(gaussian_traces) -> None:
    traces = gaussian_traces(200, seed=2, sigma=0.5)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    in_band = cal(x=1.0, sigma=0.5, reference=1.0)
    assert in_band.covered is True
    out_of_band = cal(x=1.0, sigma=0.5, reference=1.0 + 1e6)
    assert out_of_band.covered is False


def test_evaluate_hits_target_coverage(gaussian_traces) -> None:
    cal = vc.calibrate(_identity_reward, gaussian_traces(500, seed=3, sigma=0.5), alpha=0.1)
    held_out = gaussian_traces(2000, seed=4, sigma=0.5)
    report = cal.evaluate(held_out)
    assert isinstance(report, CoverageReport)
    assert abs(report.empirical_coverage - 0.9) < 0.05
    assert report.passes
    assert report.tolerance == pytest.approx(0.05)
    assert report.n == 2000
    assert report.n_in_interval == int(round(report.empirical_coverage * 2000))


def test_evaluate_tolerance_can_be_tightened(gaussian_traces) -> None:
    cal = vc.calibrate(_identity_reward, gaussian_traces(500, seed=5, sigma=0.5), alpha=0.1)
    held_out = gaussian_traces(2000, seed=6, sigma=0.5)
    loose = cal.evaluate(held_out, tolerance=0.5)
    tight = cal.evaluate(held_out, tolerance=0.0)
    assert loose.passes is True  # 50pp slack always passes
    # Empirical coverage almost never lands exactly on 0.9, so tolerance=0
    # very nearly always fails. Either outcome is acceptable provided the
    # tolerance is honoured.
    assert tight.passes == bool(abs(tight.empirical_coverage - 0.9) <= 0.0)


def test_evaluate_rejects_negative_tolerance(gaussian_traces) -> None:
    cal = vc.calibrate(_identity_reward, gaussian_traces(50, seed=7), alpha=0.1)
    with pytest.raises(ValueError, match="tolerance"):
        cal.evaluate(gaussian_traces(50, seed=8), tolerance=-0.01)


def test_calibrate_rejects_bad_alpha(gaussian_traces) -> None:
    traces = gaussian_traces(20, seed=9)
    with pytest.raises(ValueError, match="alpha"):
        vc.calibrate(_identity_reward, traces, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        vc.calibrate(_identity_reward, traces, alpha=1.0)


def test_calibrate_rejects_tiny_calibration_set(gaussian_traces) -> None:
    one = gaussian_traces(1, seed=10)
    with pytest.raises(ValueError, match="at least 2"):
        vc.calibrate(_identity_reward, one, alpha=0.1)


def test_calibrate_rejects_missing_reference_reward() -> None:
    bad_traces = [{"x": 1.0, "uncertainty": 0.5}] * 5  # missing reference_reward
    with pytest.raises(ValueError, match="reference_reward"):
        vc.calibrate(_identity_reward, bad_traces, alpha=0.1)


def test_calibrate_rejects_missing_uncertainty_for_scaled_residual() -> None:
    bad_traces = [{"x": 1.0, "reference_reward": 0.5}] * 5
    with pytest.raises(ValueError, match="uncertainty"):
        vc.calibrate(_identity_reward, bad_traces, alpha=0.1)


def test_calibrate_unknown_nonconformity_string() -> None:
    bad = [{"x": 1.0, "reference_reward": 1.0, "uncertainty": 0.5}] * 5
    with pytest.raises(ValueError, match="unknown nonconformity"):
        vc.calibrate(_identity_reward, bad, alpha=0.1, nonconformity="not_real")


def test_calibrate_invalid_nonconformity_type() -> None:
    bad = [{"x": 1.0, "reference_reward": 1.0, "uncertainty": 0.5}] * 5
    with pytest.raises(TypeError, match="must be a registered name"):
        vc.calibrate(_identity_reward, bad, alpha=0.1, nonconformity=42)  # type: ignore[arg-type]


def test_calibrate_custom_callable_nonconformity(gaussian_traces) -> None:
    def custom(trace: Mapping[str, Any], predicted: float) -> float:
        return abs(predicted - trace["reference_reward"])

    traces = gaussian_traces(200, seed=11)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity=custom)
    assert cal.nonconformity_name == "<callable>"
    # Custom callable wraps to scale-aware interval, so sigma is required.
    assert cal.is_scale_aware is True


def test_calibrate_with_nonconformity_score_object(gaussian_traces) -> None:
    traces = gaussian_traces(200, seed=12)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity=nc.SCALED_RESIDUAL)
    assert cal.nonconformity_name == "scaled_residual"


def test_quantile_increases_when_alpha_decreases(gaussian_traces) -> None:
    traces = gaussian_traces(500, seed=13)
    cal_loose = vc.calibrate(_identity_reward, traces, alpha=0.20)
    cal_tight = vc.calibrate(_identity_reward, traces, alpha=0.05)
    assert cal_tight.quantile >= cal_loose.quantile


def test_call_requires_sigma_for_scale_aware(gaussian_traces) -> None:
    traces = gaussian_traces(50, seed=14)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    with pytest.raises(ValueError, match="scale-aware"):
        cal(x=1.0)


def test_call_rejects_negative_sigma(gaussian_traces) -> None:
    traces = gaussian_traces(50, seed=15)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    with pytest.raises(ValueError, match="sigma"):
        cal(x=1.0, sigma=-0.1)


def test_abs_residual_skips_sigma_requirement() -> None:
    """abs_residual is scale-free → sigma is optional at call time."""
    rng = np.random.default_rng(17)
    traces = [
        {"x": float(rng.standard_normal()), "reference_reward": float(rng.standard_normal())}
        for _ in range(200)
    ]
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity="abs_residual")
    assert cal.is_scale_aware is False
    res = cal(x=0.0)  # no sigma required
    lo, hi = res.interval
    assert hi - lo == pytest.approx(2 * cal.quantile)


def test_binary_calibration_when_predictions_match() -> None:
    """When the calibration predictions perfectly match the references, the
    binary quantile collapses to 0 and the test-time interval is a point."""
    traces = [
        {"x": 1.0, "reference_reward": 1.0} for _ in range(200)
    ]
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity="binary")
    assert cal.quantile == pytest.approx(0.0)
    res = cal(x=1.0)
    assert res.interval == (1.0, 1.0)


def test_binary_calibration_with_disagreement_yields_vacuous_interval() -> None:
    """If references disagree often enough, the (1−α) quantile is 1 → vacuous interval."""
    rng = np.random.default_rng(20)
    traces = []
    for _ in range(200):
        x = float(rng.choice([0.0, 1.0]))
        ref = float(rng.choice([0.0, 1.0]))
        traces.append({"x": x, "reference_reward": ref})
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity="binary")
    res = cal(x=1.0)
    # alpha=0.1 with majority disagreement -> quantile lands at 1.0 -> [0,1]
    assert res.interval == (0.0, 1.0)


def test_reward_kwargs_keys_whitelist() -> None:
    def fn(*, x: float) -> float:  # only accepts x
        return float(x)

    rng = np.random.default_rng(21)
    traces = [
        {
            "x": float(rng.standard_normal()),
            "extra_unused_field": "ignored",
            "reference_reward": float(rng.standard_normal()),
            "uncertainty": 0.5,
        }
        for _ in range(50)
    ]
    # Without explicit keys, default behavior would forward "extra_unused_field"
    # and trigger TypeError. Whitelist fixes this.
    cal = vc.calibrate(fn, traces, alpha=0.1, reward_kwargs_keys=["x"])
    assert cal.reward_kwargs_keys == ("x",)


def test_reward_fn_exception_re_raised_with_index() -> None:
    def fn(*, x: float) -> float:
        if x > 0:
            raise RuntimeError("boom")
        return x

    traces = [
        {"x": -1.0, "reference_reward": 0.0, "uncertainty": 0.5},
        {"x": -1.0, "reference_reward": 0.0, "uncertainty": 0.5},
        {"x": 1.0, "reference_reward": 0.0, "uncertainty": 0.5},  # blows up
    ]
    with pytest.raises(RuntimeError, match=r"trace\[2\]"):
        vc.calibrate(fn, traces, alpha=0.1)


def test_repr_excludes_internal_fields() -> None:
    """CalibratedRewardFn repr should not leak the internal _nc_score field."""
    rng = np.random.default_rng(22)
    traces = [
        {"x": float(rng.standard_normal()), "reference_reward": float(rng.standard_normal()), "uncertainty": 0.5}
        for _ in range(20)
    ]
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    text = repr(cal)
    assert "_nc_score" not in text
    assert "quantile" in text
