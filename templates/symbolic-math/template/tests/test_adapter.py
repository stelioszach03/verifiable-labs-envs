"""Adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.data import Instance


def _toy_instance() -> Instance:
    return Instance(
        prompt="Simplify (x+1)*(x-1)",
        gold_expr="x**2 - 1",
        seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0},
    )


def test_system_prompt_mentions_env_id():
    assert "__ENV_ID__" in SYSTEM_PROMPT


def test_build_user_prompt_runs():
    inst = _toy_instance()
    text = build_user_prompt(inst)
    assert isinstance(text, str)
    assert "PROBLEM" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.prompt in text


def test_parse_response_clean_json_round_trip():
    inst = _toy_instance()
    payload = {"answer": "x**2 - 1", "confidence": 0.85}
    pred = parse_response(json.dumps(payload), inst)
    assert pred.answer_expr == "x**2 - 1"
    assert pred.confidence == pytest.approx(0.85)
    assert pred.raw == json.dumps(payload)


def test_parse_response_recovers_from_markdown_fences():
    inst = _toy_instance()
    payload = {"answer": "x + 1", "confidence": 0.5}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    pred = parse_response(fenced, inst)
    assert pred.answer_expr == "x + 1"
    assert pred.confidence == pytest.approx(0.5)


def test_parse_response_recovers_from_leading_prose():
    inst = _toy_instance()
    payload = {"answer": "y", "confidence": 0.3}
    text = "Sure, here's my answer: " + json.dumps(payload)
    pred = parse_response(text, inst)
    assert pred.answer_expr == "y"


def test_parse_response_garbage_returns_empty_prediction():
    inst = _toy_instance()
    pred = parse_response("not json at all", inst)
    assert pred.answer_expr == ""
    assert pred.confidence == 0.0


def test_parse_response_clamps_confidence_to_unit_interval():
    inst = _toy_instance()
    pred_high = parse_response(json.dumps({"answer": "x", "confidence": 1.7}), inst)
    pred_low = parse_response(json.dumps({"answer": "x", "confidence": -0.3}), inst)
    assert pred_high.confidence == 1.0
    assert pred_low.confidence == 0.0


def test_parse_response_handles_missing_confidence():
    inst = _toy_instance()
    pred = parse_response(json.dumps({"answer": "x"}), inst)
    assert pred.answer_expr == "x"
    assert pred.confidence == 0.0


def test_parse_response_handles_non_dict_json():
    inst = _toy_instance()
    pred = parse_response(json.dumps([1, 2, 3]), inst)
    assert pred.answer_expr == ""
    assert pred.confidence == 0.0
