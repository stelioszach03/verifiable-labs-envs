"""LLM adapter tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.adapter import SYSTEM_PROMPT, build_user_prompt, parse_response
from __ENV_PY__.data import SqlInstance
from __ENV_PY__.sandbox import Schema


def _toy_instance() -> SqlInstance:
    return SqlInstance(
        prompt="Toy: SELECT 1",
        template_name="toy",
        seed=0,
        schema=Schema(
            create_statements=("CREATE TABLE t (n INTEGER);",),
            seed_statements=(),
            table_names=("t",),
            column_names_by_table={"t": ("n",)},
            seed=0,
        ),
        gold_query="SELECT 1 ORDER BY 1",
        gold_query_is_ordered=True,
        gold_result_rows=((1,),),
    )


def test_system_prompt_documents_envelope():
    assert "query" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
    assert "SELECT" in SYSTEM_PROMPT


def test_build_user_prompt_includes_schema():
    inst = _toy_instance()
    text = build_user_prompt(inst)
    assert "CREATE TABLE t" in text
    assert "OUTPUT SCHEMA" in text


def test_parse_response_handles_clean_json():
    inst = _toy_instance()
    text = json.dumps({"query": "SELECT 1", "confidence": 0.7})
    pred = parse_response(text, inst)
    assert pred.query == "SELECT 1"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json():
    inst = _toy_instance()
    text = "```json\n" + json.dumps({"query": "SELECT 2", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.query == "SELECT 2"


def test_parse_response_returns_empty_on_garbage():
    inst = _toy_instance()
    pred = parse_response("totally unrelated prose", inst)
    assert pred.query == ""
    assert pred.confidence == 0.0
