"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.data import ToolCallingInstance, ToolCallingPrediction
from __ENV_PY__.reward import (
    ACTION_VALIDITY_WEIGHT,
    DEFAULT_WEIGHTS,
    STATE_MATCH_WEIGHT,
    compute_reward,
    score_components,
)
from __ENV_PY__.tools import init_state


def _toy_instance() -> ToolCallingInstance:
    return ToolCallingInstance(
        prompt="Compute 2 + 2 and submit.",
        template_name="toy",
        seed=0,
        gold_spec={"target": 4.0},
        initial_files={},
        available_tools=("calculator",),
        metadata={"alpha": 0.1, "max_tool_calls": 30},
    )


def _toy_prediction(
    *,
    tool_calls=None,
    final_text='{"answer": 4, "confidence": 0.5}',
) -> ToolCallingPrediction:
    return ToolCallingPrediction(
        tool_calls=tuple(tool_calls or []),
        final_text=final_text,
        final_state=init_state(seed=0),
        raw=final_text,
        confidence=0.5,
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_25_d6_a():
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_d2c_correctness_weights_sum_to_one():
    assert ACTION_VALIDITY_WEIGHT + STATE_MATCH_WEIGHT == pytest.approx(1.0)


def test_score_components_garbage_short_circuits():
    inst = _toy_instance()
    pred = _toy_prediction(final_text="garbage")
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_format_only_no_state_match():
    inst = _toy_instance()
    pred = _toy_prediction()
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    # Default scaffold's `_check_gold_state` returns False; the
    # action_validity term collapses to zero on an empty trajectory.
    assert components["correctness"] == 0.0


def test_compute_reward_in_unit_range():
    inst = _toy_instance()
    pred = _toy_prediction()
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert "covered" in out["meta"]


def test_compute_reward_meta_carries_action_hash():
    inst = _toy_instance()
    pred = _toy_prediction(
        tool_calls=[{"name": "calculator", "arguments": {"expression": "2 + 2"}, "result": {"value": 4.0}}]
    )
    out = compute_reward(prediction=pred, instance=inst)
    assert "action_hash" in out["meta"]
    assert len(out["meta"]["action_hash"]) == 16


def test_score_components_invalid_tool_args_zeros_parse_valid():
    inst = _toy_instance()
    bad_call = {
        "name": "calculator",
        "arguments": "not json",
        "result": {"error": "x"},
    }
    pred = _toy_prediction(
        tool_calls=[bad_call],
        final_text=json.dumps({"answer": 4}),
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
