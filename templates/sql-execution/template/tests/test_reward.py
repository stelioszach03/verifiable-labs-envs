"""Reward-function tests for __ENV_ID__."""
from __future__ import annotations

import json

import pytest

from __ENV_PY__.data import SqlInstance, SqlPrediction
from __ENV_PY__.reward import (
    DEFAULT_WEIGHTS,
    compute_reward,
    score_components,
)
from __ENV_PY__.sandbox import Schema


def _toy_instance() -> SqlInstance:
    return SqlInstance(
        prompt="Toy: SELECT 1",
        template_name="toy",
        seed=0,
        schema=Schema(
            create_statements=("CREATE TABLE t (n INTEGER);",),
            seed_statements=("INSERT INTO t (n) VALUES (1), (2), (3);",),
            table_names=("t",),
            column_names_by_table={"t": ("n",)},
            seed=0,
        ),
        gold_query="SELECT n FROM t ORDER BY n ASC",
        gold_query_is_ordered=True,
        gold_result_rows=((1,), (2,), (3,)),
        metadata={"alpha": 0.1, "max_rows": 100, "timeout_s": 2.0},
    )


def _toy_prediction(query: str) -> SqlPrediction:
    return SqlPrediction(
        query=query,
        raw=json.dumps({"query": query, "confidence": 0.5}),
        confidence=0.5,
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_26_d7_a():
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_score_components_garbage_short_circuits():
    inst = _toy_instance()
    pred = SqlPrediction(query="", raw="not json", confidence=0.0)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_dml_zeros_parse_valid():
    inst = _toy_instance()
    pred = _toy_prediction("DROP TABLE t")
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0


def test_score_components_gold_query_clears_correctness():
    inst = _toy_instance()
    pred = _toy_prediction(inst.gold_query)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == pytest.approx(1.0)


def test_compute_reward_with_conformal_emits_covered_flag():
    inst = _toy_instance()
    pred = _toy_prediction(inst.gold_query)
    out = compute_reward(
        prediction=pred, instance=inst,
        timeout_s=2.0, conformal_quantile=0.5,
    )
    meta = out["meta"]
    assert "covered" in meta
    assert isinstance(meta["covered"], bool)
    assert "residual" in meta
    assert 0.0 <= meta["residual"] <= 1.0


def test_compute_reward_carries_query_and_schema_hashes():
    inst = _toy_instance()
    pred = _toy_prediction(inst.gold_query)
    out = compute_reward(prediction=pred, instance=inst, timeout_s=2.0)
    assert "schema_hash" in out["meta"]
    assert "query_hash" in out["meta"]
