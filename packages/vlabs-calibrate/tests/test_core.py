"""Tests for the split-conformal primitives in vlabs_calibrate.core.

These mirror the legacy tests in ``tests/test_conformal.py`` of the
``verifiable-labs-envs`` repo plus a parity check that the two
implementations produce identical numerical output when both are
installed (the legacy module is the upstream source of truth).
"""
from __future__ import annotations

import numpy as np
import pytest

from vlabs_calibrate import core


def test_split_conformal_quantile_basic() -> None:
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = core.split_conformal_quantile(residuals, alpha=0.1)
    # ceil(11 * 0.9) / 10 = 10 / 10 = 1.0 -> the max residual
    assert q == pytest.approx(1.0)


def test_split_conformal_quantile_alpha_sweep_monotone() -> None:
    rng = np.random.default_rng(0)
    residuals = np.abs(rng.standard_normal(500))
    qs = [core.split_conformal_quantile(residuals, alpha=a) for a in (0.2, 0.1, 0.05)]
    assert qs[0] <= qs[1] <= qs[2]  # tighter alpha -> larger quantile


def test_split_conformal_quantile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        core.split_conformal_quantile(np.array([]), alpha=0.1)


def test_split_conformal_quantile_rejects_bad_alpha() -> None:
    residuals = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        core.split_conformal_quantile(residuals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        core.split_conformal_quantile(residuals, alpha=1.0)


def test_scaled_residuals_basic() -> None:
    x_hat = np.array([1.0, 2.0, 3.0])
    x_true = np.array([0.0, 3.0, 3.0])
    sigma_hat = np.array([1.0, 2.0, 1.0])
    r = core.scaled_residuals(x_hat, x_true, sigma_hat)
    np.testing.assert_allclose(r, [1.0, 0.5, 0.0], atol=1e-6)


def test_scaled_residuals_rejects_shape_mismatch() -> None:
    x_hat = np.array([1.0, 2.0])
    x_true = np.array([1.0, 2.0, 3.0])
    sigma_hat = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        core.scaled_residuals(x_hat, x_true, sigma_hat)


def test_scaled_residuals_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        core.scaled_residuals(
            np.array([1.0]), np.array([0.0]), np.array([-0.1])
        )


def test_scaled_residuals_eps_floor() -> None:
    # sigma=0 entries don't divide-by-zero thanks to the eps floor.
    x_hat = np.array([1.0])
    x_true = np.array([0.0])
    sigma_hat = np.array([0.0])
    r = core.scaled_residuals(x_hat, x_true, sigma_hat, eps=1e-8)
    assert np.isfinite(r[0])
    assert r[0] > 0


def test_interval_basic() -> None:
    x_hat = np.array([1.0, 2.0])
    sigma_hat = np.array([0.5, 1.0])
    lo, hi = core.interval(x_hat, sigma_hat, q=2.0)
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [2.0, 4.0])


def test_interval_rejects_negative_q() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        core.interval(np.array([1.0]), np.array([1.0]), q=-0.5)


def test_coverage_basic() -> None:
    x_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 2.5, 2.5, 3.5])
    upper = np.array([2.0, 3.0, 3.5, 5.0])
    # in, out (2.0 < 2.5), in (3.0 in [2.5, 3.5]), in (4.0 in [3.5, 5.0])
    assert core.coverage(x_true, lower, upper) == pytest.approx(0.75)


def test_coverage_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        core.coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([2.0]))


def test_coverage_score_peaks_at_target() -> None:
    target = 0.9
    assert core.coverage_score(target, target) == pytest.approx(1.0)
    assert core.coverage_score(target - 0.1, target) == pytest.approx(0.9)
    assert core.coverage_score(target + 0.1, target) == pytest.approx(0.9)
    assert core.coverage_score(0.0, target) == pytest.approx(0.1)


def test_coverage_score_clamped_to_zero() -> None:
    # |observed - target| > 1 cannot happen for valid coverage values, but
    # the function must still clip at zero.
    assert core.coverage_score(0.0, 0.0) == pytest.approx(1.0)
    assert core.coverage_score(1.0, 0.0) == pytest.approx(0.0)


def test_coverage_score_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="target"):
        core.coverage_score(0.5, target=-0.1)
    with pytest.raises(ValueError, match="target"):
        core.coverage_score(0.5, target=1.1)


def test_parity_with_verifiable_labs_envs_conformal() -> None:
    """When both packages are installed, the two implementations agree exactly."""
    legacy = pytest.importorskip("verifiable_labs_envs.conformal")

    rng = np.random.default_rng(123)
    residuals = np.abs(rng.standard_normal(500))
    for alpha in (0.05, 0.1, 0.2):
        assert core.split_conformal_quantile(residuals, alpha) == pytest.approx(
            legacy.split_conformal_quantile(residuals, alpha)
        )

    x_hat = rng.standard_normal(64)
    x_true = rng.standard_normal(64)
    sigma_hat = np.abs(rng.standard_normal(64)) + 0.1
    np.testing.assert_allclose(
        core.scaled_residuals(x_hat, x_true, sigma_hat),
        legacy.scaled_residuals(x_hat, x_true, sigma_hat),
    )

    lo_a, hi_a = core.interval(x_hat, sigma_hat, q=1.5)
    lo_b, hi_b = legacy.interval(x_hat, sigma_hat, q=1.5)
    np.testing.assert_allclose(lo_a, lo_b)
    np.testing.assert_allclose(hi_a, hi_b)

    lower = x_hat - 1.0 * sigma_hat
    upper = x_hat + 1.0 * sigma_hat
    assert core.coverage(x_true, lower, upper) == pytest.approx(
        legacy.coverage(x_true, lower, upper)
    )
    assert core.coverage_score(0.85, target=0.9) == pytest.approx(
        legacy.coverage_score(0.85, target=0.9)
    )
