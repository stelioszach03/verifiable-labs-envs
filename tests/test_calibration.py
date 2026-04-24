"""Tests for the auto_calibrate utility (sprint-giga Task 0)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from verifiable_labs_envs.calibration import ConformalConfig, auto_calibrate


@dataclass
class _Instance:
    x_true: np.ndarray
    seed: int


@dataclass
class _Prediction:
    x_hat: np.ndarray
    sigma_hat: np.ndarray


def _generate(seed: int) -> _Instance:
    rng = np.random.default_rng(seed)
    return _Instance(x_true=rng.standard_normal(4), seed=seed)


def _good_baseline(instance: _Instance) -> _Prediction:
    """Baseline that returns the truth ± small noise (high-quality)."""
    rng = np.random.default_rng(instance.seed + 1000)
    noise = 0.1 * rng.standard_normal(4)
    return _Prediction(
        x_hat=instance.x_true + noise,
        sigma_hat=np.full(4, 0.2),
    )


def _bad_baseline(instance: _Instance) -> _Prediction:
    """Baseline that returns zeros (high non-conformity)."""
    return _Prediction(
        x_hat=np.zeros_like(instance.x_true),
        sigma_hat=np.full(4, 0.2),
    )


def test_auto_calibrate_returns_config():
    cfg = auto_calibrate(_generate, _good_baseline, n_calibration=30, alpha=0.1)
    assert isinstance(cfg, ConformalConfig)
    assert cfg.n_calibration == 30
    assert cfg.alpha == 0.1
    assert cfg.quantile > 0.0


def test_quantile_increases_when_baseline_is_worse():
    cfg_good = auto_calibrate(_generate, _good_baseline, n_calibration=50, alpha=0.1)
    cfg_bad = auto_calibrate(_generate, _bad_baseline, n_calibration=50, alpha=0.1)
    assert cfg_bad.quantile > cfg_good.quantile


def test_quantile_decreases_as_alpha_increases():
    cfg_low_alpha = auto_calibrate(_generate, _good_baseline, n_calibration=50, alpha=0.05)
    cfg_high_alpha = auto_calibrate(_generate, _good_baseline, n_calibration=50, alpha=0.30)
    assert cfg_low_alpha.quantile >= cfg_high_alpha.quantile


def test_rejects_tiny_n_calibration():
    with pytest.raises(ValueError, match="n_calibration"):
        auto_calibrate(_generate, _good_baseline, n_calibration=1, alpha=0.1)


def test_rejects_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        auto_calibrate(_generate, _good_baseline, n_calibration=30, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        auto_calibrate(_generate, _good_baseline, n_calibration=30, alpha=1.0)


def test_custom_non_conformity_fn_used():
    called = []

    def my_nc(pred, inst):
        called.append((pred, inst))
        return 0.42

    cfg = auto_calibrate(
        _generate, _good_baseline, n_calibration=10, alpha=0.1,
        non_conformity_fn=my_nc,
    )
    assert len(called) == 10
    assert cfg.quantile == pytest.approx(0.42)


def test_config_carries_mean_and_std():
    cfg = auto_calibrate(_generate, _good_baseline, n_calibration=30, alpha=0.1)
    assert cfg.non_conformity_mean > 0.0
    assert cfg.non_conformity_std >= 0.0
