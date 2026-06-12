"""LLM adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.corpus import Corpus, Document, NeedleAnchor
from __ENV_PY__.data import NeedleInstance


def _toy_instance() -> NeedleInstance:
    return NeedleInstance(
        question="What is the secret code?",
        template_name="toy",
        seed=0,
        corpus=Corpus(
            documents=(
                Document(id=0, title="Doc A", body="Body of doc A."),
                Document(id=1, title="Doc B", body="Body of doc B."),
            ),
            seed=0,
        ),
        needle_text="The secret code is XYZ-1234.",
        needle_anchor=NeedleAnchor(
            document_id=0, char_offset=0,
            needle_text="The secret code is XYZ-1234.",
            is_distractor=False,
        ),
        position_mode="start",
    )


def test_system_prompt_documents_envelope():
    assert "answer" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_build_user_prompt_includes_documents_and_question():
    inst = _toy_instance()
    text = build_user_prompt(inst)
    assert "QUESTION:" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.question in text
    assert "---DOCUMENT 0:" in text


def test_parse_response_handles_clean_json():
    inst = _toy_instance()
    text = json.dumps({"answer": "XYZ-1234", "confidence": 0.7})
    pred = parse_response(text, inst)
    assert pred.answer == "XYZ-1234"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json():
    inst = _toy_instance()
    text = "```json\n" + json.dumps({"answer": "ABC", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.answer == "ABC"


def test_parse_response_returns_empty_on_garbage():
    inst = _toy_instance()
    pred = parse_response("totally unrelated prose", inst)
    assert pred.answer == ""
    assert pred.confidence == 0.0
