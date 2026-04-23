"""Unit tests for the sparse-Fourier environment.

Every ISTA-heavy test uses explicit calibration quantiles and small hyperparameters
so the full suite runs in a few seconds on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.envs import sparse_fourier as sf
from verifiable_labs_envs.forward_ops import sparse_fourier_forward

# ---------- Generation ----------


def test_generate_instance_determinism() -> None:
    a = sf.generate_instance(seed=123)
    b = sf.generate_instance(seed=123)
    np.testing.assert_array_equal(a.x_true, b.x_true)
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.y, b.y)


def test_generate_instance_different_seeds_differ() -> None:
    a = sf.generate_instance(seed=1)
    b = sf.generate_instance(seed=2)
    assert not np.array_equal(a.x_true, b.x_true)


def test_generate_instance_sparsity_and_shapes() -> None:
    inst = sf.generate_instance(seed=0, n=128, m=32, k=5)
    assert inst.x_true.shape == (128,)
    assert inst.y.shape == (32,)
    assert inst.mask.shape == (32,)
    assert inst.support_true.shape == (5,)
    nonzero = np.where(inst.x_true != 0.0)[0]
    assert nonzero.shape == (5,)
    np.testing.assert_array_equal(np.sort(inst.support_true), nonzero)


def test_forward_matches_measurements_up_to_noise() -> None:
    inst = sf.generate_instance(seed=42, sigma=0.01)
    y_clean = sparse_fourier_forward(inst.x_true, inst.mask)
    resid = inst.y - y_clean
    # per-entry variance of resid should be close to sigma^2 (real+imag together)
    empirical_std = float(np.std(resid))
    assert empirical_std < 0.05  # loose bound around sigma=0.01


def test_generate_instance_rejects_k_gt_n() -> None:
    with pytest.raises(ValueError, match="k"):
        sf.generate_instance(seed=0, n=4, k=10)


def test_as_inputs_hides_oracle_fields() -> None:
    inst = sf.generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "x_true" not in inputs
    assert "support_true" not in inputs
    assert set(inputs) == {"y", "mask", "sigma", "n", "k"}


# ---------- Baselines ----------


def test_ista_baseline_beats_zero_on_default() -> None:
    inst = sf.generate_instance(seed=3)
    env = sf.SparseFourierEnv(conformal_quantile=3.0)  # arbitrary; we only look at NMSE

    zero = sf.zero_baseline(**inst.as_inputs())
    ista = sf.ista_baseline(**inst.as_inputs(), n_iters=150, n_bootstrap=8, seed=3)

    s_zero = env.score(zero, inst)
    s_ista = env.score(ista, inst)

    assert s_ista["components"]["nmse"] > 0.7
    # zero baseline gives NMSE = 1 exactly -> score_nmse = exp(-1/0.5) ~= 0.135
    assert s_zero["components"]["nmse"] < 0.15
    assert s_ista["components"]["support"] > 0.5
    assert s_ista["reward"] > s_zero["reward"]


def test_ista_baseline_shapes_and_positivity() -> None:
    inst = sf.generate_instance(seed=5)
    pred = sf.ista_baseline(**inst.as_inputs(), n_iters=80, n_bootstrap=5, seed=5)
    assert pred.x_hat.shape == (inst.n,)
    assert pred.sigma_hat.shape == (inst.n,)
    assert np.all(pred.sigma_hat > 0)
    assert pred.support_hat is not None
    assert pred.support_hat.shape == (inst.k,)


# ---------- Scoring ----------


def test_score_components_are_in_unit_interval() -> None:
    inst = sf.generate_instance(seed=9)
    env = sf.SparseFourierEnv(conformal_quantile=2.5)
    pred = sf.ista_baseline(**inst.as_inputs(), n_iters=80, n_bootstrap=5, seed=9)
    out = env.score(pred, inst)
    for key, value in out["components"].items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["target_coverage"] == pytest.approx(0.9)


def test_score_reports_conformal_quantile_in_meta() -> None:
    inst = sf.generate_instance(seed=0)
    env = sf.SparseFourierEnv(conformal_quantile=1.25)
    pred = sf.zero_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    assert out["meta"]["conformal_quantile"] == pytest.approx(1.25)


def test_score_respects_custom_weights() -> None:
    inst = sf.generate_instance(seed=0)
    env = sf.SparseFourierEnv(conformal_quantile=3.0)
    pred = sf.ista_baseline(**inst.as_inputs(), n_iters=80, n_bootstrap=5, seed=0)
    out_default = env.score(pred, inst)
    out_nmse_only = env.score(pred, inst, weights={"nmse": 1.0, "support": 0.0, "conformal": 0.0})
    assert out_nmse_only["reward"] == pytest.approx(out_default["components"]["nmse"])


def test_support_f1_falls_back_to_topk_when_support_not_provided() -> None:
    inst = sf.generate_instance(seed=0, k=4)
    # prediction with no explicit support, but x_hat large exactly on the true support
    x_hat = np.zeros(inst.n, dtype=np.float64)
    x_hat[inst.support_true] = inst.x_true[inst.support_true]
    pred = sf.Prediction(
        x_hat=x_hat,
        sigma_hat=np.ones(inst.n, dtype=np.float64),
        support_hat=None,
    )
    env = sf.SparseFourierEnv(conformal_quantile=1.0)
    out = env.score(pred, inst)
    assert out["components"]["support"] == pytest.approx(1.0)


# ---------- Calibration + factory ----------


def test_calibrate_conformal_quantile_positive_and_finite() -> None:
    q = sf.calibrate_conformal_quantile(
        n_samples=8, alpha=0.1, n_bootstrap=3, n_iters=50
    )
    assert np.isfinite(q)
    assert q > 0


def test_load_environment_with_explicit_quantile_skips_calibration() -> None:
    env = sf.load_environment(calibration_quantile=2.0)
    assert env.conformal_quantile == pytest.approx(2.0)


def test_load_environment_fast_caches() -> None:
    sf._cached_quantile.cache_clear()
    env_a = sf.load_environment(fast=True)
    env_b = sf.load_environment(fast=True)
    assert env_a.conformal_quantile == env_b.conformal_quantile
    info = sf._cached_quantile.cache_info()
    assert info.hits >= 1  # second call hit the cache


def test_env_run_baseline_returns_full_dict() -> None:
    env = sf.load_environment(calibration_quantile=3.0)
    out = env.run_baseline(seed=0, n_iters=80, n_bootstrap=5)
    assert set(out) == {"reward", "components", "meta"}
    assert set(out["components"]) == {"nmse", "support", "conformal"}


@pytest.mark.slow
def test_calibrated_environment_achieves_target_coverage() -> None:
    """On the same distribution used for calibration, empirical coverage should be near 1-alpha."""
    q = sf.calibrate_conformal_quantile(n_samples=20, alpha=0.1, n_bootstrap=5, n_iters=80)
    env = sf.SparseFourierEnv(conformal_quantile=q)
    covs = []
    for seed in range(20, 30):
        inst = env.generate_instance(seed)
        pred = sf.ista_baseline(**inst.as_inputs(), n_iters=80, n_bootstrap=5, seed=seed)
        out = env.score(pred, inst)
        covs.append(out["meta"]["coverage"])
    empirical = float(np.mean(covs))
    assert 0.75 <= empirical <= 1.0, f"empirical coverage {empirical:.2f} far from target 0.90"
