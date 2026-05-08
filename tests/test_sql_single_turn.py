"""Tests for the ``sql-single-turn`` env (Phase 26.B)."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.envs.sql_single_turn import (
    DEFAULT_HYPERPARAMS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    SqlPrediction,
    SqlSingleTurnEnv,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    load_environment,
    parse_response,
    score_components,
)

# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "sql-single-turn"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_26_d7_a() -> None:
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_default_hyperparams_carry_alpha_and_caps() -> None:
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert DEFAULT_HYPERPARAMS["max_rows"] == 10_000
    assert DEFAULT_HYPERPARAMS["timeout_s"] == 10.0
    assert DEFAULT_HYPERPARAMS["max_query_bytes"] == 32 * 1024


# ── Procedural lattice ────────────────────────────────────────────────


def test_generate_instance_is_deterministic_per_seed() -> None:
    a = generate_instance(seed=11)
    b = generate_instance(seed=11)
    assert a.template_name == b.template_name
    assert a.gold_query == b.gold_query
    assert a.gold_result_rows == b.gold_result_rows


def test_generate_instance_carries_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    assert inst.gold_query
    assert inst.gold_result_rows
    assert inst.schema.create_statements
    assert inst.schema.seed_statements


def test_as_inputs_excludes_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_query" not in inputs
    assert "gold_result_rows" not in inputs
    assert "prompt" in inputs
    assert "schema" in inputs


def test_as_inputs_does_not_leak_seed_inserts() -> None:
    """Public schema view must not expose INSERT data."""
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    schema = inputs["schema"]
    # CREATE TABLE statements only — no INSERTs in the public surface.
    for stmt in schema["create_statements"]:
        assert "INSERT" not in stmt.upper()


def test_user_prompt_does_not_leak_gold_query() -> None:
    """Sweep seeds: the gold query string must not appear in the
    rendered user prompt verbatim."""
    for seed in range(0, 30, 3):
        inst = generate_instance(seed=seed)
        rendered = build_user_prompt(inst)
        # Gold query mustn't appear in the prompt body (the prompt
        # only carries the natural-language question + schema).
        assert inst.gold_query not in rendered


def test_user_prompt_carries_schema() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    for tbl in inst.schema.table_names:
        assert tbl in text
    assert "OUTPUT SCHEMA" in text


# ── Reward kernel — pure short-circuits ──────────────────────────────


def _toy_prediction(query: str) -> SqlPrediction:
    return SqlPrediction(
        query=query,
        raw=json.dumps({"query": query, "confidence": 0.5}),
        confidence=0.5,
    )


def test_score_components_baseline_zero_everywhere() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_garbled_text_short_circuits() -> None:
    inst = generate_instance(seed=0)
    pred = SqlPrediction(query="", raw="not json", confidence=0.0)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0


def test_score_components_dml_query_zeroes_parse_valid() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction("DROP TABLE products")
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_random_function_zeroes_correctness() -> None:
    """A query that uses RANDOM() passes the read-only gate but
    fails the determinism check inside the sandbox."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction("SELECT RANDOM() FROM products ORDER BY 1")
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    # Sandbox rejects RANDOM() before execution → correctness = 0.
    assert components["correctness"] == 0.0


def test_score_components_gold_query_clears_correctness() -> None:
    """The procedural gold query scores 1.0 on every seed."""
    for seed in range(0, 24, 3):
        inst = generate_instance(seed=seed)
        pred = _toy_prediction(inst.gold_query)
        components = score_components(pred, inst, timeout_s=5.0)
        assert components["format_valid"] == 1.0, f"seed {seed}"
        assert components["parse_valid"] == 1.0, f"seed {seed}"
        assert components["correctness"] == pytest.approx(1.0), (
            f"gold did not match for seed={seed} "
            f"template={inst.template_name}"
        )


def test_score_components_wrong_query_scores_partial() -> None:
    """A SELECT that returns the wrong rows scores format + parse, not correctness."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction("SELECT 1 ORDER BY 1")
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == 0.0


# ── Reward aggregation ─────────────────────────────────────────────


def test_compute_reward_in_unit_range_and_emits_meta() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["template"] == inst.template_name
    assert "schema_hash" in out["meta"]
    assert "covered" in out["meta"]


def test_compute_reward_carries_query_hash() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_query)
    out = compute_reward(prediction=pred, instance=inst)
    assert "query_hash" in out["meta"]
    assert len(out["meta"]["query_hash"]) == 16


def test_compute_reward_cache_key_stable() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_query)
    a = compute_reward(prediction=pred, instance=inst)
    b = compute_reward(prediction=pred, instance=inst)
    assert a["meta"]["cache_key"] == b["meta"]["cache_key"]


# ── Adapter ────────────────────────────────────────────────────────


def test_system_prompt_documents_envelope_and_constraints() -> None:
    assert "query" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
    assert "SELECT" in SYSTEM_PROMPT


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    payload = json.dumps({"query": "SELECT 1", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert pred.query == "SELECT 1"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = "```json\n" + json.dumps({"query": "SELECT 2", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.query == "SELECT 2"


def test_parse_response_returns_empty_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose", inst)
    assert pred.query == ""
    assert pred.confidence == 0.0


def test_parse_response_clamps_confidence() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response(json.dumps({"query": "SELECT 1", "confidence": 5.0}), inst)
    assert pred.confidence == pytest.approx(1.0)
    pred = parse_response(json.dumps({"query": "SELECT 1", "confidence": -1.0}), inst)
    assert pred.confidence == pytest.approx(0.0)


# ── Env class ──────────────────────────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, SqlSingleTurnEnv)
    assert env.name == NAME


def test_env_score_returns_canonical_dict() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = env.score(pred, inst)
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert "covered" in out["meta"]


def test_env_run_baseline_finite_reward() -> None:
    env = load_environment(calibration_quantile=0.5)
    out = env.run_baseline(seed=0)
    assert isinstance(out["reward"], float)
    assert 0.0 <= out["reward"] <= 1.0


def test_env_score_with_gold_query_full_credit() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_query)
    out = env.score(pred, inst)
    assert out["reward"] == pytest.approx(1.0)
