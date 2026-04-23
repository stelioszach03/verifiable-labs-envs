"""Unit tests for the simplified low-dose CT environment."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.envs import lodopab_ct as ct

# ---------- Instance generation ----------


def test_generate_instance_shapes() -> None:
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    n_side = inst.shape[0]
    assert inst.x_true.shape == (n_side, n_side)
    assert inst.y.shape == (n_side, inst.n_angles)
    assert inst.angles_deg.shape == (inst.n_angles,)


def test_generate_instance_deterministic() -> None:
    a = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    b = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    np.testing.assert_array_equal(a.x_true, b.x_true)
    np.testing.assert_array_equal(a.y, b.y)


def test_generate_instance_different_noise_per_seed() -> None:
    a = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    b = ct.generate_instance(seed=1, phantom_name="shepp_logan")
    np.testing.assert_array_equal(a.x_true, b.x_true)
    assert not np.array_equal(a.y, b.y)


def test_generate_instance_rotation_uses_seed() -> None:
    a = ct.generate_instance(seed=0)
    b = ct.generate_instance(seed=1)
    # Different seeds -> different phantoms in the rotation
    assert a.phantom_name != b.phantom_name or not np.array_equal(a.x_true, b.x_true)


def test_generate_instance_rejects_unknown_phantom() -> None:
    with pytest.raises(KeyError):
        ct.generate_instance(seed=0, phantom_name="not_a_real_phantom_name")


def test_angles_are_equispaced_in_half_circle() -> None:
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan", n_angles=30)
    assert inst.angles_deg[0] == 0.0
    assert inst.angles_deg[-1] < 180.0
    diffs = np.diff(inst.angles_deg)
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)


def test_as_inputs_hides_oracle() -> None:
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    inputs = inst.as_inputs()
    assert "x_true" not in inputs
    assert "phantom_name" not in inputs


# ---------- Baselines ----------


def test_fbp_baseline_shapes_and_positivity() -> None:
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    pred = ct.fbp_baseline(**inst.as_inputs())
    assert pred.x_hat.shape == inst.shape
    assert pred.sigma_hat.shape == inst.shape
    assert np.all(pred.sigma_hat > 0)


def test_fbp_beats_zero_baseline_on_psnr_and_ssim() -> None:
    env = ct.LodopabCtEnv(conformal_quantile=3.0)
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    fbp = ct.fbp_baseline(**inst.as_inputs())
    zero = ct.zero_baseline(**inst.as_inputs())
    s_fbp = env.score(fbp, inst)
    s_zero = env.score(zero, inst)
    assert s_fbp["components"]["psnr"] > s_zero["components"]["psnr"]
    assert s_fbp["components"]["ssim"] > s_zero["components"]["ssim"]
    assert s_fbp["reward"] > s_zero["reward"]


def test_fbp_has_spatially_varying_sigma_hat() -> None:
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    pred = ct.fbp_baseline(**inst.as_inputs())
    assert pred.sigma_hat.std() > 0.001


# ---------- Scoring ----------


def test_score_components_in_unit_interval() -> None:
    env = ct.LodopabCtEnv(conformal_quantile=3.0)
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    pred = ct.fbp_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    for key, value in out["components"].items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"
    assert 0.0 <= out["reward"] <= 1.0


def test_score_perfect_reconstruction_gets_top_psnr() -> None:
    env = ct.LodopabCtEnv(conformal_quantile=0.1)
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    pred = ct.Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.full(inst.shape, 0.01, dtype=np.float64),
    )
    out = env.score(pred, inst)
    assert out["components"]["psnr"] == pytest.approx(1.0)
    assert out["components"]["ssim"] > 0.99


def test_score_respects_custom_weights() -> None:
    env = ct.LodopabCtEnv(conformal_quantile=3.0)
    inst = ct.generate_instance(seed=0, phantom_name="shepp_logan")
    pred = ct.fbp_baseline(**inst.as_inputs())
    out = env.score(pred, inst, weights={"psnr": 1.0, "ssim": 0.0, "conformal": 0.0})
    assert out["reward"] == pytest.approx(out["components"]["psnr"])


# ---------- Calibration + factory ----------


def test_calibrate_conformal_quantile_positive_and_finite() -> None:
    q = ct.calibrate_conformal_quantile(alpha=0.1, n_samples=2)
    assert np.isfinite(q)
    assert q > 0


def test_load_environment_with_explicit_quantile() -> None:
    env = ct.load_environment(calibration_quantile=2.5)
    assert env.conformal_quantile == pytest.approx(2.5)


def test_load_environment_fast_caches() -> None:
    ct._cached_quantile.cache_clear()
    env_a = ct.load_environment(fast=True)
    env_b = ct.load_environment(fast=True)
    assert env_a.conformal_quantile == env_b.conformal_quantile
    assert ct._cached_quantile.cache_info().hits >= 1


def test_env_run_baseline_returns_full_dict() -> None:
    env = ct.load_environment(calibration_quantile=3.0)
    out = env.run_baseline(seed=0, phantom_name="shepp_logan")
    assert set(out) == {"reward", "components", "meta"}
    assert set(out["components"]) == {"psnr", "ssim", "conformal"}
