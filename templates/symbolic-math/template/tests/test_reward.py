"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import json

from __ENV_PY__.data import Instance, Prediction
from __ENV_PY__.reward import (
    DEFAULT_WEIGHTS,
    compute_reward,
    score_components,
)


def _toy_instance(prompt: str, gold_expr: str) -> Instance:
    return Instance(
        prompt=prompt,
        gold_expr=gold_expr,
        seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0},
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == 1.0


def test_score_components_perfect_match_returns_one():
    inst = _toy_instance(prompt="Simplify (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="x**2 - 1",
        raw=json.dumps({"answer": "x**2 - 1", "confidence": 0.9}),
        confidence=0.9,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correct"] == 1.0


def test_score_components_handles_equivalent_form():
    """`(x-1)*(x+1)` and `x**2 - 1` are equivalent — simplify catches it."""
    inst = _toy_instance(prompt="Expand (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="(x-1)*(x+1)",
        raw=json.dumps({"answer": "(x-1)*(x+1)", "confidence": 0.8}),
        confidence=0.8,
    )
    components = score_components(pred, inst)
    assert components["correct"] == 1.0


def test_score_components_wrong_answer_zero_correct():
    inst = _toy_instance(prompt="Simplify (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="x**2 + 1",  # wrong sign
        raw=json.dumps({"answer": "x**2 + 1", "confidence": 0.9}),
        confidence=0.9,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correct"] == 0.0


def test_score_components_unparseable_short_circuits():
    inst = _toy_instance(prompt="Simplify x", gold_expr="x")
    pred = Prediction(
        answer_expr="this is not sympy",
        raw=json.dumps({"answer": "this is not sympy", "confidence": 0.5}),
        confidence=0.5,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
    assert components["correct"] == 0.0


def test_score_components_bad_json_short_circuits_to_zero():
    inst = _toy_instance(prompt="Simplify x", gold_expr="x")
    pred = Prediction(
        answer_expr="x",
        raw="not json at all",
        confidence=0.5,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correct"] == 0.0


def test_compute_reward_scalar_in_range():
    inst = _toy_instance(prompt="Simplify x", gold_expr="x")
    pred = Prediction(
        answer_expr="x",
        raw=json.dumps({"answer": "x", "confidence": 0.5}),
        confidence=0.5,
    )
    out = compute_reward(prediction=pred, instance=inst)
    assert 0.0 <= out["reward"] <= 1.0
    assert "components" in out
    assert "meta" in out


def test_compute_reward_with_conformal_emits_covered_flag():
    inst = _toy_instance(prompt="Simplify x", gold_expr="x")
    pred = Prediction(
        answer_expr="x",
        raw=json.dumps({"answer": "x", "confidence": 0.5}),
        confidence=0.5,
    )
    out = compute_reward(
        prediction=pred,
        instance=inst,
        conformal_quantile=0.5,
    )
    assert "covered" in out["meta"]
    assert isinstance(out["meta"]["covered"], bool)
    assert "residual" in out["meta"]
    assert 0.0 <= out["meta"]["residual"] <= 1.0


def test_simplify_timeout_returns_no_credit():
    """A pathological gold expression that triggers a slow simplify
    must time out and score zero on `correct`, not wedge the test."""
    # Construct a deeply nested expression that can stall simplify.
    # 10ms timeout forces a fast bail.
    inst = _toy_instance(
        prompt="meta-test",
        gold_expr="(x**100 + y**100 + z**100) * (sin(x) + cos(y))**5",
    )
    pred = Prediction(
        answer_expr="0",  # wrong, and a non-trivial diff to simplify
        raw=json.dumps({"answer": "0", "confidence": 0.1}),
        confidence=0.1,
    )
    components = score_components(pred, inst, timeout_s=0.01)
    # parse_valid should still succeed; correct must be 0 either by
    # genuine inequality or by timeout — both score 0.
    assert components["parse_valid"] == 1.0
    assert components["correct"] == 0.0
