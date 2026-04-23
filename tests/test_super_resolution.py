"""Unit tests for the super-resolution environment."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.envs import super_resolution as sr
from verifiable_labs_envs.forward_ops import blur_downsample

# ---------- Instance generation ----------


def test_generate_instance_shapes() -> None:
    inst = sr.generate_instance(seed=0, image_name="camera")
    h, w = inst.shape
    f = inst.factor
    assert inst.x_true.shape == (h, w)
    assert inst.y.shape == (h // f, w // f)
    assert inst.x_true.min() >= 0.0 and inst.x_true.max() <= 1.0


def test_generate_instance_image_rotation_is_deterministic() -> None:
    a = sr.generate_instance(seed=0)
    b = sr.generate_instance(seed=0)
    assert a.image_name == b.image_name
    np.testing.assert_array_equal(a.x_true, b.x_true)
    np.testing.assert_array_equal(a.y, b.y)


def test_generate_instance_different_noise_per_seed() -> None:
    # Same image via explicit name, different noise draws from seed
    a = sr.generate_instance(seed=0, image_name="camera")
    b = sr.generate_instance(seed=1, image_name="camera")
    np.testing.assert_array_equal(a.x_true, b.x_true)  # same HR image
    assert not np.array_equal(a.y, b.y)  # different noise


def test_generate_instance_measurement_close_to_clean_forward() -> None:
    inst = sr.generate_instance(seed=0, image_name="camera", noise_sigma=0.001)
    y_clean = blur_downsample(inst.x_true, inst.blur_sigma, inst.factor)
    resid_std = float(np.std(inst.y - y_clean))
    assert resid_std < 0.01


def test_generate_instance_rejects_unknown_image() -> None:
    with pytest.raises(KeyError):
        sr.generate_instance(seed=0, image_name="definitely_not_a_real_image")


def test_as_inputs_hides_oracle() -> None:
    inst = sr.generate_instance(seed=0, image_name="camera")
    inputs = inst.as_inputs()
    assert "x_true" not in inputs
    assert "image_name" not in inputs
    assert set(inputs) >= {"y", "shape", "blur_sigma", "factor", "noise_sigma"}


# ---------- Baselines ----------


def test_bicubic_baseline_shapes_and_positivity() -> None:
    inst = sr.generate_instance(seed=0, image_name="camera")
    pred = sr.bicubic_baseline(**inst.as_inputs())
    assert pred.x_hat.shape == inst.shape
    assert pred.sigma_hat.shape == inst.shape
    assert np.all(pred.sigma_hat > 0)
    # bicubic can overshoot [0, 1] slightly at edges; allow a small margin
    assert pred.x_hat.min() >= -0.1 and pred.x_hat.max() <= 1.1


def test_bicubic_baseline_beats_zero_baseline_on_psnr() -> None:
    env = sr.SuperResolutionEnv(conformal_quantile=3.0)
    inst = sr.generate_instance(seed=0, image_name="camera")
    bicubic = sr.bicubic_baseline(**inst.as_inputs())
    zero = sr.zero_baseline(**inst.as_inputs())
    s_bi = env.score(bicubic, inst)
    s_zero = env.score(zero, inst)
    assert s_bi["components"]["psnr"] > s_zero["components"]["psnr"]
    assert s_bi["components"]["ssim"] > s_zero["components"]["ssim"]
    assert s_bi["reward"] > s_zero["reward"]


def test_bicubic_baseline_has_spatially_varying_sigma_hat() -> None:
    inst = sr.generate_instance(seed=0, image_name="camera")
    pred = sr.bicubic_baseline(**inst.as_inputs())
    # Should not be constant -- gradient term produces variation
    assert pred.sigma_hat.std() > 0.01


# ---------- Scoring ----------


def test_score_components_in_unit_interval() -> None:
    env = sr.SuperResolutionEnv(conformal_quantile=3.0)
    inst = sr.generate_instance(seed=0, image_name="camera")
    pred = sr.bicubic_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    for key, value in out["components"].items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"
    assert 0.0 <= out["reward"] <= 1.0


def test_score_psnr_db_in_meta_is_finite_and_positive() -> None:
    env = sr.SuperResolutionEnv(conformal_quantile=3.0)
    inst = sr.generate_instance(seed=0, image_name="camera")
    pred = sr.bicubic_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    assert np.isfinite(out["meta"]["psnr_db"])
    assert out["meta"]["psnr_db"] > 0


def test_score_perfect_reconstruction_gets_top_psnr() -> None:
    env = sr.SuperResolutionEnv(conformal_quantile=0.1)  # tight interval
    inst = sr.generate_instance(seed=0, image_name="camera")
    # Perfect prediction -- x_hat == x_true, tiny sigma_hat
    pred = sr.Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.full(inst.shape, 0.01, dtype=np.float64),
    )
    out = env.score(pred, inst)
    assert out["components"]["psnr"] == pytest.approx(1.0)
    assert out["components"]["ssim"] > 0.99


def test_score_respects_custom_weights() -> None:
    env = sr.SuperResolutionEnv(conformal_quantile=3.0)
    inst = sr.generate_instance(seed=0, image_name="camera")
    pred = sr.bicubic_baseline(**inst.as_inputs())
    out_psnr_only = env.score(pred, inst, weights={"psnr": 1.0, "ssim": 0.0, "conformal": 0.0})
    assert out_psnr_only["reward"] == pytest.approx(out_psnr_only["components"]["psnr"])


# ---------- Calibration + factory ----------


def test_calibrate_conformal_quantile_positive_and_finite() -> None:
    q = sr.calibrate_conformal_quantile(alpha=0.1, n_samples=2)
    assert np.isfinite(q)
    assert q > 0


def test_load_environment_with_explicit_quantile_skips_calibration() -> None:
    env = sr.load_environment(calibration_quantile=2.5)
    assert env.conformal_quantile == pytest.approx(2.5)


def test_load_environment_fast_caches() -> None:
    sr._cached_quantile.cache_clear()
    env_a = sr.load_environment(fast=True)
    env_b = sr.load_environment(fast=True)
    assert env_a.conformal_quantile == env_b.conformal_quantile
    assert sr._cached_quantile.cache_info().hits >= 1


def test_env_run_baseline_returns_full_dict() -> None:
    env = sr.load_environment(calibration_quantile=3.0)
    out = env.run_baseline(seed=0, image_name="camera")
    assert set(out) == {"reward", "components", "meta"}
    assert set(out["components"]) == {"psnr", "ssim", "conformal"}


@pytest.mark.slow
def test_calibrated_environment_achieves_target_coverage_near_0p9() -> None:
    q = sr.calibrate_conformal_quantile(alpha=0.1, n_samples=3)
    env = sr.SuperResolutionEnv(conformal_quantile=q)
    covs = []
    for seed, name in enumerate(["moon", "coffee", "chelsea"]):
        inst = env.generate_instance(seed + 100, image_name=name)
        pred = sr.bicubic_baseline(**inst.as_inputs())
        covs.append(env.score(pred, inst)["meta"]["coverage"])
    empirical = float(np.mean(covs))
    assert 0.8 <= empirical <= 1.0, f"empirical coverage {empirical:.2f} far from target 0.9"
