"""Adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import numpy as np
import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.env import generate_instance


def test_system_prompt_mentions_env_id():
    assert "__ENV_ID__" in SYSTEM_PROMPT


def test_build_user_prompt_runs():
    """Skipped while ground-truth/forward stubs raise."""
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_instance not yet implemented")
    text = build_user_prompt(inst)
    assert isinstance(text, str)
    assert "INPUTS" in text
    assert "OUTPUT SCHEMA" in text


def test_parse_response_handles_truth_round_trip():
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_instance not yet implemented")
    truth_payload = {
        "x_hat_x1000": [int(round(float(v) * 1000)) for v in
                        np.asarray(inst.x_true).ravel().tolist()],
        "sigma_hat_x1000": [10] * inst.x_true.size,
    }
    pred = parse_response(json.dumps(truth_payload), inst)
    assert pred.x_hat.shape == inst.x_true.shape
    assert pred.sigma_hat.shape == inst.x_true.shape
    np.testing.assert_allclose(pred.x_hat, inst.x_true, atol=1e-3)


def test_parse_response_garbage_returns_zero_prediction():
    try:
        inst = generate_instance(seed=0)
    except NotImplementedError:
        pytest.skip("generate_instance not yet implemented")
    pred = parse_response("not json at all", inst)
    np.testing.assert_array_equal(pred.x_hat, np.zeros_like(inst.x_true))
