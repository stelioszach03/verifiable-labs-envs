"""LLM adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.data import ToolCallingInstance


def _toy_instance() -> ToolCallingInstance:
    return ToolCallingInstance(
        prompt="Compute 2 + 2 and submit.",
        template_name="toy",
        seed=0,
        gold_spec={"target": 4.0},
        initial_files={"note.txt": "remember to add"},
        available_tools=("calculator", "send_message"),
    )


def test_system_prompt_documents_envelope():
    assert "answer" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_build_user_prompt_lists_tools_and_workspace():
    inst = _toy_instance()
    text = build_user_prompt(inst)
    assert "AVAILABLE TOOLS" in text
    for tool in inst.available_tools:
        assert tool in text
    assert "note.txt" in text


def test_parse_response_handles_clean_json():
    inst = _toy_instance()
    text = json.dumps({"answer": 4, "confidence": 0.7})
    pred = parse_response(text, inst)
    assert pred.confidence == pytest.approx(0.7)
    assert pred.tool_calls == ()


def test_parse_response_handles_fenced_json():
    inst = _toy_instance()
    text = "```json\n" + json.dumps({"answer": "x", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.confidence == pytest.approx(0.4)


def test_parse_response_returns_zero_confidence_on_garbage():
    inst = _toy_instance()
    pred = parse_response("garbage", inst)
    assert pred.confidence == 0.0
