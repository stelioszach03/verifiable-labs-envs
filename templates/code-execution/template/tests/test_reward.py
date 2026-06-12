"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import json
import sys

import pytest

from __ENV_PY__.data import CodeInstance, CodePrediction
from __ENV_PY__.reward import (
    DEFAULT_WEIGHTS,
    compute_reward,
    score_components,
)


def _toy_instance() -> CodeInstance:
    """A tiny hard-coded instance the reward tests use directly.

    Bypasses ``generate_problem`` (which may still be NotImplemented in
    a fresh scaffold) so the reward kernel can be exercised end-to-end.
    """
    return CodeInstance(
        function_signature="def f(x: int) -> int:",
        docstring="Return x doubled.",
        visible_tests=("f(2) == 4",),
        hidden_tests=("f(0) == 0", "f(-3) == -6"),
        gold_solution="def f(x: int) -> int:\n    return x * 2",
        template_name="toy",
        seed=0,
        metadata={"alpha": 0.1, "sandbox_timeout_s": 5.0},
    )


def _toy_prediction(code: str) -> CodePrediction:
    return CodePrediction(
        code=code,
        raw=json.dumps({"code": code, "confidence": 0.5}),
        confidence=0.5,
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_24_d7c():
    """D7-C ruling: 0.10 format + 0.20 parse + 0.70 pass_rate."""
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["pass_rate"] == pytest.approx(0.70)


def test_score_components_bad_json_short_circuits_to_zero():
    if sys.platform != "linux":
        pytest.skip("sandbox primitive requires Linux")
    inst = _toy_instance()
    pred = CodePrediction(code="", raw="not json", confidence=0.0)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["pass_rate"] == 0.0


def test_score_components_uncompileable_short_circuits():
    if sys.platform != "linux":
        pytest.skip("sandbox primitive requires Linux")
    inst = _toy_instance()
    pred = _toy_prediction("def broken(:\n  pass")  # syntax error
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
    assert components["pass_rate"] == 0.0


def test_score_components_gold_solution_clears_all_tests():
    if sys.platform != "linux":
        pytest.skip("sandbox primitive requires Linux")
    inst = _toy_instance()
    pred = _toy_prediction(inst.gold_solution)
    components = score_components(pred, inst, timeout_s=10.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["pass_rate"] == pytest.approx(1.0)


def test_compute_reward_with_conformal_emits_covered_flag():
    if sys.platform != "linux":
        pytest.skip("sandbox primitive requires Linux")
    inst = _toy_instance()
    pred = _toy_prediction(inst.gold_solution)
    out = compute_reward(
        prediction=pred,
        instance=inst,
        timeout_s=10.0,
        conformal_quantile=0.5,
    )
    meta = out["meta"]
    assert "covered" in meta
    assert isinstance(meta["covered"], bool)
    assert "residual" in meta
    assert 0.0 <= meta["residual"] <= 1.0
