"""Unit tests for the conformal utilities."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs import conformal


def test_split_conformal_quantile_basic() -> None:
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = conformal.split_conformal_quantile(residuals, alpha=0.1)
    # ceil(11 * 0.9) / 10 = 10 / 10 = 1.0; so returns the max residual
    assert q == pytest.approx(1.0)


def test_split_conformal_quantile_alpha_sweep_monotone() -> None:
    rng = np.random.default_rng(0)
    residuals = np.abs(rng.standard_normal(500))
    qs = [conformal.split_conformal_quantile(residuals, alpha=a) for a in (0.2, 0.1, 0.05)]
    assert qs[0] <= qs[1] <= qs[2]  # tighter alpha => larger quantile


def test_split_conformal_quantile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        conformal.split_conformal_quantile(np.array([]), alpha=0.1)


def test_split_conformal_quantile_rejects_bad_alpha() -> None:
    residuals = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        conformal.split_conformal_quantile(residuals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        conformal.split_conformal_quantile(residuals, alpha=1.0)


def test_scaled_residuals_basic() -> None:
    x_hat = np.array([1.0, 2.0, 3.0])
    x_true = np.array([0.0, 3.0, 3.0])
    sigma_hat = np.array([1.0, 2.0, 1.0])
    r = conformal.scaled_residuals(x_hat, x_true, sigma_hat)
    np.testing.assert_allclose(r, [1.0, 0.5, 0.0], atol=1e-6)


def test_scaled_residuals_rejects_shape_mismatch() -> None:
    x_hat = np.array([1.0, 2.0])
    x_true = np.array([1.0, 2.0, 3.0])
    sigma_hat = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        conformal.scaled_residuals(x_hat, x_true, sigma_hat)


def test_scaled_residuals_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        conformal.scaled_residuals(
            np.array([1.0]), np.array([0.0]), np.array([-0.1])
        )


def test_interval_basic() -> None:
    x_hat = np.array([1.0, 2.0])
    sigma_hat = np.array([0.5, 1.0])
    lo, hi = conformal.interval(x_hat, sigma_hat, q=2.0)
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [2.0, 4.0])


def test_interval_rejects_negative_q() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        conformal.interval(np.array([1.0]), np.array([1.0]), q=-0.5)


def test_coverage_basic() -> None:
    x_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 2.5, 2.5, 3.5])
    upper = np.array([2.0, 3.0, 3.5, 5.0])
    # in, out (2.0 < 2.5), in (3.0 in [2.5, 3.5]), in (4.0 in [3.5, 5.0])
    assert conformal.coverage(x_true, lower, upper) == pytest.approx(0.75)


def test_coverage_score_peaks_at_target() -> None:
    assert conformal.coverage_score(0.9, target=0.9) == pytest.approx(1.0)
    assert conformal.coverage_score(0.8, target=0.9) == pytest.approx(0.9)
    assert conformal.coverage_score(1.0, target=0.9) == pytest.approx(0.9)
    assert conformal.coverage_score(0.0, target=0.9) == pytest.approx(0.1)


def test_coverage_score_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="target"):
        conformal.coverage_score(0.5, target=1.5)
