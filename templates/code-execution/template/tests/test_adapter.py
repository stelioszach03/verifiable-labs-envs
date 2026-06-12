"""LLM adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.data import CodeInstance


def _toy_instance() -> CodeInstance:
    return CodeInstance(
        function_signature="def f(x: int) -> int:",
        docstring="Return x doubled.",
        visible_tests=("f(2) == 4",),
        hidden_tests=("f(0) == 0",),
        gold_solution="def f(x): return x * 2",
        template_name="toy",
        seed=0,
    )


def test_system_prompt_documents_json_envelope():
    assert "code" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_build_user_prompt_includes_signature_and_visible_tests():
    inst = _toy_instance()
    text = build_user_prompt(inst)
    assert inst.function_signature in text
    assert "OUTPUT SCHEMA" in text


def test_parse_response_handles_clean_json():
    inst = _toy_instance()
    payload = json.dumps({"code": "def f(x): return x * 2", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert pred.code == "def f(x): return x * 2"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json():
    inst = _toy_instance()
    text = (
        "Sure, here's my answer:\n"
        "```json\n"
        '{"code": "def f(x): return x*2", "confidence": 0.4}\n'
        "```"
    )
    pred = parse_response(text, inst)
    assert pred.code == "def f(x): return x*2"


def test_parse_response_returns_empty_on_garbage():
    inst = _toy_instance()
    pred = parse_response("totally unrelated prose without a JSON block", inst)
    assert pred.code == ""
    assert pred.confidence == 0.0
