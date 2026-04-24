"""Tests for the MRI-knee-reconstruction env (sprint-giga Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs.mri_knee import (
    MRIKneeEnv,
    Prediction,
    _load_ground_truth_image,
    _psnr_db,
    _psnr_score,
    generate_instance,
    tv_regularized_baseline,
    zero_filled_baseline,
)
from verifiable_labs_envs.envs.mri_knee_multiturn import (
    MRIKneeMultiturnEnv,
)
from verifiable_labs_envs.envs.mri_knee_multiturn import (
    load_environment as load_mt,
)
from verifiable_labs_envs.solvers.adapters.mri_knee import MRIKneeLLMAdapter
from verifiable_labs_envs.solvers.adapters.mri_knee_multiturn import (
    MRIKneeMultiturnAdapter,
)

ENV_NAME = "mri-knee-reconstruction"
MT_ENV_NAME = "mri-knee-reconstruction-multiturn"


def test_mri_envs_registered():
    assert ENV_NAME in list_environments()
    assert MT_ENV_NAME in list_environments()


def test_load_env_returns_mri_env():
    env = load_environment(ENV_NAME, calibration_quantile=2.0)
    assert isinstance(env, MRIKneeEnv)


def test_load_env_returns_mt_env():
    env = load_environment(MT_ENV_NAME, calibration_quantile=2.0)
    assert isinstance(env, MRIKneeMultiturnEnv)


def test_ground_truth_image_is_normalized_and_right_shape():
    img = _load_ground_truth_image(seed=0, shape=(16, 16))
    assert img.shape == (16, 16)
    assert 0.0 <= img.min() <= img.max() <= 1.0


def test_generate_instance_shapes():
    inst = generate_instance(seed=0)
    h, w = inst.shape
    assert inst.x_true.shape == (h, w)
    assert inst.y.shape == (h, w)
    assert inst.mask.shape == (h, w)
    assert inst.zero_filled.shape == (h, w)


def test_generate_instance_determinism():
    a = generate_instance(seed=42)
    b = generate_instance(seed=42)
    assert np.array_equal(a.x_true, b.x_true)


def test_generate_instance_mask_undersamples():
    inst = generate_instance(seed=0)
    keep_frac = float(inst.mask.sum()) / float(inst.mask.size)
    assert 0.15 < keep_frac < 0.50  # 4x acceleration ~ 25% keep


def test_mask_dc_column_is_kept():
    """Sanity: DC-aligned center means column 0 is always in the low-freq set."""
    inst = generate_instance(seed=0)
    # In numpy FFT convention, DC is at (0, 0).
    assert inst.mask[0, 0] == 1.0


def test_zero_filled_baseline_produces_image_in_01():
    inst = generate_instance(seed=0)
    pred = zero_filled_baseline(**inst.as_inputs())
    assert pred.x_hat.shape == inst.shape
    assert pred.x_hat.min() >= 0.0 and pred.x_hat.max() <= 1.0


def test_tv_baseline_runs_and_produces_image():
    """TV baseline should at least run; convergence is known-buggy in v1."""
    inst = generate_instance(seed=0)
    pred = tv_regularized_baseline(**inst.as_inputs(), n_iters=10)
    assert pred.x_hat.shape == inst.shape


def test_psnr_score_mapping():
    assert _psnr_score(15.0) == pytest.approx(0.0)
    assert _psnr_score(35.0) == pytest.approx(1.0)
    assert _psnr_score(25.0) == pytest.approx(0.5)


def test_psnr_db_high_for_perfect_recovery():
    x = np.random.default_rng(0).uniform(0, 1, size=(16, 16))
    db = _psnr_db(x, x.copy())
    assert db >= 50.0  # near-infinite for identical arrays


def test_perfect_prediction_scores_high():
    env = MRIKneeEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    pred = Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.full(inst.shape, 0.05),
    )
    out = env.score(pred, inst)
    assert out["reward"] > 0.8


def test_components_in_0_1():
    env = MRIKneeEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    pred = zero_filled_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    for name, val in out["components"].items():
        assert 0.0 <= val <= 1.0, f"component {name}={val} out of range"


def test_zero_filled_run_baseline_convenience():
    env = MRIKneeEnv(conformal_quantile=2.0)
    out = env.run_baseline(seed=0)
    assert "reward" in out


def test_adapter_builds_prompt_with_zero_filled_grid():
    adapter = MRIKneeLLMAdapter()
    inst = generate_instance(seed=0)
    prompt = adapter.build_user_prompt(inst)
    assert "zero_filled" in prompt
    assert "image" in prompt


def test_adapter_parses_truth_image():
    import json
    adapter = MRIKneeLLMAdapter()
    inst = generate_instance(seed=0)
    x_u8 = (np.clip(inst.x_true, 0, 1) * 255).round().astype(int).tolist()
    truth_json = json.dumps({"image": x_u8})
    pred = adapter.parse_response(truth_json, inst)
    assert pred.x_hat.shape == inst.shape


def test_mt_adapter_builds_followup_with_residual():
    import json
    adapter = MRIKneeMultiturnAdapter()
    inst = generate_instance(seed=0)
    x_u8 = (np.clip(inst.x_true, 0, 1) * 255).round().astype(int).tolist()
    truth_json = json.dumps({"image": x_u8})
    pred = adapter.parse_response(truth_json, inst)
    followup = adapter.build_followup_turn([], pred, inst)
    assert "residual_l2_x1000" in followup
    assert "residual_mag_u8_peak_normalized" in followup


def test_mt_env_rejects_zero_max_turns():
    with pytest.raises(ValueError):
        MRIKneeMultiturnEnv(conformal_quantile=2.0, max_turns=0)


def test_mt_load_environment_factory():
    env = load_mt(calibration_quantile=2.0, max_turns=2)
    assert env.max_turns == 2
