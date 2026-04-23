"""Unit tests for the per-environment LLM adapters. No network."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verifiable_labs_envs.envs import lodopab_ct as ct
from verifiable_labs_envs.envs import sparse_fourier as sf
from verifiable_labs_envs.envs import super_resolution as sr
from verifiable_labs_envs.forward_ops import radon_fbp
from verifiable_labs_envs.solvers import FakeLLMSolver, LLMSolverError
from verifiable_labs_envs.solvers.adapters._common import extract_json_block
from verifiable_labs_envs.solvers.adapters.lodopab_ct import (
    COARSE_SIZE,
    LodopabCtLLMAdapter,
)
from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.adapters.super_resolution import SuperResolutionLLMAdapter

# ---------- extract_json_block ----------


def test_extract_json_plain() -> None:
    assert extract_json_block('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence() -> None:
    text = '```json\n{"support_idx": [1, 2]}\n```'
    assert extract_json_block(text) == {"support_idx": [1, 2]}


def test_extract_json_with_prose_before_and_after() -> None:
    text = 'Sure, here it is: {"a": 1, "b": [2, 3]}. Hope that helps!'
    assert extract_json_block(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_rejects_empty() -> None:
    with pytest.raises(LLMSolverError, match="empty"):
        extract_json_block("")


def test_extract_json_rejects_non_object() -> None:
    # Plain array is valid JSON but not an object
    with pytest.raises(LLMSolverError, match="no JSON block"):
        extract_json_block("[1, 2, 3]")


def test_extract_json_rejects_unbalanced_braces() -> None:
    with pytest.raises(LLMSolverError, match="unbalanced"):
        extract_json_block('{"a": 1, "b": ')


# ---------- Sparse-Fourier adapter ----------


@pytest.fixture
def sf_instance() -> sf.Instance:
    return sf.generate_instance(seed=0)


def test_sf_adapter_registered() -> None:
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert "sparse-fourier-recovery" in _ADAPTERS
    assert isinstance(_ADAPTERS["sparse-fourier-recovery"], SparseFourierLLMAdapter)


def test_sf_build_user_prompt_contains_all_fields(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    prompt = adapter.build_user_prompt(sf_instance)
    assert "INPUTS:" in prompt
    assert "OUTPUT SCHEMA:" in prompt
    assert '"n": 256' in prompt
    assert '"k": 10' in prompt
    assert '"mask":' in prompt
    assert '"y_re_x1000":' in prompt
    assert '"y_im_x1000":' in prompt


def test_sf_parse_valid_response(sf_instance: sf.Instance) -> None:
    """Feed the ground-truth support back through the parser."""
    adapter = SparseFourierLLMAdapter()
    support = sf_instance.support_true
    amps = sf_instance.x_true[support]
    payload = {
        "support_idx": [int(i) for i in support],
        "support_amp_x1000": [int(round(v * 1000)) for v in amps],
    }
    pred = adapter.parse_response(json.dumps(payload), sf_instance)
    np.testing.assert_array_equal(np.sort(pred.support_hat), np.sort(support))
    np.testing.assert_allclose(pred.x_hat[support], amps, atol=2e-3)
    assert np.all(pred.sigma_hat > 0)


def test_sf_parse_with_markdown_fence(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {
        "support_idx": [int(i) for i in sf_instance.support_true],
        "support_amp_x1000": [0] * sf_instance.k,
    }
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    pred = adapter.parse_response(wrapped, sf_instance)
    assert pred.support_hat is not None


def test_sf_parse_rejects_wrong_length(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {"support_idx": [1, 2, 3], "support_amp_x1000": [1, 2, 3]}
    with pytest.raises(LLMSolverError, match="expected 10 entries"):
        adapter.parse_response(json.dumps(payload), sf_instance)


def test_sf_parse_rejects_out_of_range_index(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {
        "support_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 999],
        "support_amp_x1000": [0] * sf_instance.k,
    }
    with pytest.raises(LLMSolverError, match="out of range"):
        adapter.parse_response(json.dumps(payload), sf_instance)


def test_sf_parse_rejects_duplicate_indices(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {
        "support_idx": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "support_amp_x1000": [0] * sf_instance.k,
    }
    with pytest.raises(LLMSolverError, match="duplicate"):
        adapter.parse_response(json.dumps(payload), sf_instance)


def test_sf_parse_rejects_missing_key(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {"support_idx": [0] * sf_instance.k}
    with pytest.raises(LLMSolverError, match="support_amp_x1000"):
        adapter.parse_response(json.dumps(payload), sf_instance)


def test_sf_parse_rejects_non_integer_index(sf_instance: sf.Instance) -> None:
    adapter = SparseFourierLLMAdapter()
    payload = {
        "support_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, "abc"],
        "support_amp_x1000": [0] * sf_instance.k,
    }
    with pytest.raises(LLMSolverError, match="non-integer"):
        adapter.parse_response(json.dumps(payload), sf_instance)


def test_sf_end_to_end_with_fake_solver(sf_instance: sf.Instance) -> None:
    support = sf_instance.support_true
    amps = sf_instance.x_true[support]
    response = json.dumps({
        "support_idx": [int(i) for i in support],
        "support_amp_x1000": [int(round(v * 1000)) for v in amps],
    })
    solver = FakeLLMSolver(response)
    pred = solver.solve("sparse-fourier-recovery", sf_instance)
    env = sf.SparseFourierEnv(conformal_quantile=2.0)
    result = env.score(pred, sf_instance)
    assert result["components"]["nmse"] > 0.9  # fed the truth -> near-perfect
    assert result["components"]["support"] == pytest.approx(1.0)


# ---------- Super-resolution adapter ----------


@pytest.fixture
def sr_instance() -> sr.Instance:
    return sr.generate_instance(seed=0, image_name="camera")


def test_sr_adapter_registered() -> None:
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert "super-resolution-div2k-x4" in _ADAPTERS
    assert isinstance(_ADAPTERS["super-resolution-div2k-x4"], SuperResolutionLLMAdapter)


def test_sr_build_user_prompt_contains_image(sr_instance: sr.Instance) -> None:
    adapter = SuperResolutionLLMAdapter()
    prompt = adapter.build_user_prompt(sr_instance)
    assert "INPUTS:" in prompt
    assert "OUTPUT SCHEMA:" in prompt
    assert '"image":' in prompt
    assert f'"n_rows":{sr_instance.y.shape[0]}' in prompt.replace(" ", "")


def test_sr_parse_valid_response(sr_instance: sr.Instance) -> None:
    adapter = SuperResolutionLLMAdapter()
    # Feed the LR measurement right back as a uint8 grid
    y = np.clip(sr_instance.y, 0.0, 1.0)
    y_u8 = (y * 255.0).round().astype(int).tolist()
    payload = {"image": y_u8}
    pred = adapter.parse_response(json.dumps(payload), sr_instance)
    assert pred.x_hat.shape == sr_instance.shape
    assert pred.sigma_hat.shape == sr_instance.shape
    assert np.all(pred.sigma_hat > 0)


def test_sr_parse_rejects_wrong_row_count(sr_instance: sr.Instance) -> None:
    adapter = SuperResolutionLLMAdapter()
    payload = {"image": [[0] * sr_instance.y.shape[1]] * 3}  # too few rows
    with pytest.raises(LLMSolverError, match="expected"):
        adapter.parse_response(json.dumps(payload), sr_instance)


def test_sr_parse_rejects_pixel_out_of_range(sr_instance: sr.Instance) -> None:
    adapter = SuperResolutionLLMAdapter()
    lr_rows, lr_cols = sr_instance.y.shape
    grid = [[0] * lr_cols for _ in range(lr_rows)]
    grid[0][0] = 500
    payload = {"image": grid}
    with pytest.raises(LLMSolverError, match="out of range"):
        adapter.parse_response(json.dumps(payload), sr_instance)


def test_sr_parse_rejects_missing_key(sr_instance: sr.Instance) -> None:
    adapter = SuperResolutionLLMAdapter()
    with pytest.raises(LLMSolverError, match="image"):
        adapter.parse_response('{"something_else": [[1]]}', sr_instance)


def test_sr_end_to_end_with_fake_solver(sr_instance: sr.Instance) -> None:
    # Feed back the LR image itself -> after bicubic upsample, same quality as bicubic baseline
    y_u8 = (np.clip(sr_instance.y, 0.0, 1.0) * 255.0).round().astype(int).tolist()
    solver = FakeLLMSolver(json.dumps({"image": y_u8}))
    pred = solver.solve("super-resolution-div2k-x4", sr_instance)
    env = sr.SuperResolutionEnv(conformal_quantile=2.0)
    result = env.score(pred, sr_instance)
    # Should beat the zero baseline
    zero = sr.zero_baseline(**sr_instance.as_inputs())
    zero_result = env.score(zero, sr_instance)
    assert result["components"]["psnr"] > zero_result["components"]["psnr"]


# ---------- LoDoPaB-CT adapter ----------


@pytest.fixture
def ct_instance() -> ct.Instance:
    return ct.generate_instance(seed=0, phantom_name="shepp_logan")


def test_ct_adapter_registered() -> None:
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert "lodopab-ct-simplified" in _ADAPTERS
    assert isinstance(_ADAPTERS["lodopab-ct-simplified"], LodopabCtLLMAdapter)


def test_ct_build_user_prompt_contains_image(ct_instance: ct.Instance) -> None:
    adapter = LodopabCtLLMAdapter()
    prompt = adapter.build_user_prompt(ct_instance)
    assert "INPUTS" in prompt
    assert "OUTPUT SCHEMA:" in prompt
    assert '"image":' in prompt


def test_ct_parse_valid_response(ct_instance: ct.Instance) -> None:
    adapter = LodopabCtLLMAdapter()
    # Feed the adapter's own rough FBP back through
    fbp = radon_fbp(
        ct_instance.y, ct_instance.angles_deg, output_size=ct_instance.shape[0]
    )
    fbp = np.clip(fbp, 0.0, 1.0)
    from skimage.transform import resize

    coarse = resize(fbp, (COARSE_SIZE, COARSE_SIZE), order=3, anti_aliasing=True)
    coarse_u8 = (coarse * 255.0).round().astype(int).tolist()
    payload = {"image": coarse_u8}
    pred = adapter.parse_response(json.dumps(payload), ct_instance)
    assert pred.x_hat.shape == ct_instance.shape
    assert pred.sigma_hat.shape == ct_instance.shape


def test_ct_parse_rejects_wrong_shape(ct_instance: ct.Instance) -> None:
    adapter = LodopabCtLLMAdapter()
    payload = {"image": [[0] * COARSE_SIZE] * 8}  # too few rows
    with pytest.raises(LLMSolverError, match="expected"):
        adapter.parse_response(json.dumps(payload), ct_instance)


def test_ct_parse_rejects_pixel_out_of_range(ct_instance: ct.Instance) -> None:
    adapter = LodopabCtLLMAdapter()
    grid = [[0] * COARSE_SIZE for _ in range(COARSE_SIZE)]
    grid[5][5] = 999
    with pytest.raises(LLMSolverError, match="out of range"):
        adapter.parse_response(json.dumps({"image": grid}), ct_instance)


def test_ct_end_to_end_with_fake_solver(ct_instance: ct.Instance) -> None:
    # The fake solver replies with all zeros -> sanity: pipeline completes, returns valid score
    grid = [[0] * COARSE_SIZE for _ in range(COARSE_SIZE)]
    solver = FakeLLMSolver(json.dumps({"image": grid}))
    pred = solver.solve("lodopab-ct-simplified", ct_instance)
    env = ct.LodopabCtEnv(conformal_quantile=2.0)
    result = env.score(pred, ct_instance)
    assert 0.0 <= result["reward"] <= 1.0
    assert "psnr" in result["components"]
