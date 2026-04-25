"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import numpy as np
import pytest

from __ENV_PY__.data import Instance, Prediction
from __ENV_PY__.reward import (
    coverage_score,
    empirical_coverage,
    score_point_estimate,
)


def _toy_instance(x_true: np.ndarray) -> Instance:
    return Instance(
        y=np.zeros_like(x_true),
        x_true=x_true,
        seed=0,
        metadata={"alpha": 0.1, "noise_sigma": 0.05},
    )


def test_score_point_estimate_perfect_returns_one():
    x = np.array([1.0, -2.0, 0.5])
    pred = Prediction(x_hat=x.copy(), sigma_hat=np.full(3, 0.1))
    inst = _toy_instance(x)
    s = score_point_estimate(pred, inst)
    assert s == pytest.approx(1.0)


def test_score_point_estimate_far_off_decays():
    x = np.array([1.0, 2.0, 3.0])
    pred = Prediction(x_hat=np.zeros(3), sigma_hat=np.ones(3))
    inst = _toy_instance(x)
    s = score_point_estimate(pred, inst)
    assert 0.0 <= s < 0.2


def test_empirical_coverage_full_cover():
    x = np.array([1.0, -1.0, 0.0])
    pred = Prediction(x_hat=x.copy(), sigma_hat=np.full(3, 1.0))
    inst = _toy_instance(x)
    cov = empirical_coverage(pred, inst, conformal_quantile=2.0)
    assert cov == pytest.approx(1.0)


def test_empirical_coverage_zero_when_too_narrow():
    x = np.array([1.0, -1.0, 0.0])
    pred = Prediction(x_hat=np.zeros(3), sigma_hat=np.full(3, 0.001))
    inst = _toy_instance(x)
    cov = empirical_coverage(pred, inst, conformal_quantile=0.001)
    assert cov < 1.0


def test_coverage_score_peaks_at_target():
    target = 0.9
    on_target = coverage_score(0.9, target)
    high = coverage_score(1.0, target)
    low = coverage_score(0.5, target)
    assert on_target > high
    assert on_target > low
    assert 0.0 <= on_target <= 1.0
