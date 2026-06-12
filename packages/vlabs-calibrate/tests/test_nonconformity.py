"""Tests for vlabs_calibrate.nonconformity — built-in scores + registry."""
from __future__ import annotations

import pytest

from vlabs_calibrate import nonconformity as nc


def test_registered_names_includes_builtins() -> None:
    names = nc.registered_names()
    assert "scaled_residual" in names
    assert "abs_residual" in names
    assert "binary" in names


def test_get_returns_score_object() -> None:
    s = nc.get("scaled_residual")
    assert isinstance(s, nc.NonconformityScore)
    assert s.name == "scaled_residual"
    assert s.is_scale_aware is True


def test_get_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown nonconformity"):
        nc.get("not_a_real_score")


def test_scaled_residual_score_basic() -> None:
    trace = {"reference_reward": 1.0, "uncertainty": 0.5}
    s = nc.SCALED_RESIDUAL.score_fn(trace, predicted=2.0, eps=1e-8)
    assert s == pytest.approx(2.0)  # |2 - 1| / 0.5


def test_scaled_residual_eps_floor_protects_zero_sigma() -> None:
    trace = {"reference_reward": 1.0, "uncertainty": 0.0}
    s = nc.SCALED_RESIDUAL.score_fn(trace, predicted=2.0, eps=1e-2)
    assert s == pytest.approx(100.0)  # 1 / 0.01


def test_scaled_residual_interval_uses_sigma() -> None:
    lo, hi = nc.SCALED_RESIDUAL.interval_fn(predicted=1.0, sigma=0.5, quantile=2.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(2.0)


def test_abs_residual_ignores_sigma_in_interval() -> None:
    trace = {"reference_reward": 0.5}
    assert nc.ABS_RESIDUAL.score_fn(trace, predicted=1.0, eps=1e-8) == pytest.approx(0.5)
    lo_a, hi_a = nc.ABS_RESIDUAL.interval_fn(predicted=1.0, sigma=10.0, quantile=0.3)
    lo_b, hi_b = nc.ABS_RESIDUAL.interval_fn(predicted=1.0, sigma=0.0, quantile=0.3)
    assert (lo_a, hi_a) == (lo_b, hi_b)
    assert (lo_a, hi_a) == pytest.approx((0.7, 1.3))
    assert nc.ABS_RESIDUAL.is_scale_aware is False


def test_binary_score_is_zero_or_one() -> None:
    trace = {"reference_reward": 1.0}
    assert nc.BINARY.score_fn(trace, predicted=1.0, eps=1e-8) == 0.0
    assert nc.BINARY.score_fn(trace, predicted=0.0, eps=1e-8) == 1.0


def test_binary_interval_collapses_when_quantile_is_zero() -> None:
    lo, hi = nc.BINARY.interval_fn(predicted=1.0, sigma=0.0, quantile=0.0)
    assert lo == hi == pytest.approx(1.0)


def test_binary_interval_is_vacuous_at_quantile_one() -> None:
    lo, hi = nc.BINARY.interval_fn(predicted=1.0, sigma=0.0, quantile=1.0)
    assert (lo, hi) == (0.0, 1.0)


def test_register_rejects_duplicate_without_overwrite() -> None:
    custom = nc.NonconformityScore(
        name="scaled_residual",  # already registered
        required_trace_keys=("reference_reward",),
        score_fn=lambda t, p, eps: 0.0,
        interval_fn=lambda p, s, q: (p, p),
        is_scale_aware=False,
    )
    with pytest.raises(ValueError, match="already registered"):
        nc.register(custom)


def test_register_then_get_roundtrip() -> None:
    custom = nc.NonconformityScore(
        name="_test_custom_score",
        required_trace_keys=("reference_reward",),
        score_fn=lambda t, p, eps: 0.5,
        interval_fn=lambda p, s, q: (p - 1, p + 1),
        is_scale_aware=False,
    )
    nc.register(custom)
    try:
        retrieved = nc.get("_test_custom_score")
        assert retrieved is custom
    finally:
        # Clean up so other tests don't see the leftover registration.
        from vlabs_calibrate.nonconformity import _REGISTRY  # noqa: PLC0415
        _REGISTRY.pop("_test_custom_score", None)
