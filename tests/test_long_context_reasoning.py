"""Tests for the ``long-context-reasoning`` env (Phase 27.D)."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.envs.long_context_reasoning import (
    DEFAULT_HYPERPARAMS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    TEMPLATE_NAMES,
    LongContextReasoningEnv,
    ReasoningPrediction,
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
    assert NAME == "long-context-reasoning"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_27_d7_a() -> None:
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_default_hyperparams_carry_alpha_and_caps() -> None:
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert "target_tokens" in DEFAULT_HYPERPARAMS
    assert "document_count" in DEFAULT_HYPERPARAMS


def test_template_roster_locked() -> None:
    """D9 — three multi-hop templates ship in v0.0.1."""
    assert set(TEMPLATE_NAMES) == {
        "chain_two_hop",
        "chain_three_hop",
        "arithmetic_over_facts",
    }


# ── Procedural lattice ────────────────────────────────────────────────


def test_generate_instance_is_deterministic_per_seed() -> None:
    a = generate_instance(seed=11)
    b = generate_instance(seed=11)
    assert a.template_name == b.template_name
    assert a.gold_answer == b.gold_answer
    assert a.gold_chain_doc_ids == b.gold_chain_doc_ids
    assert a.distractor_doc_ids == b.distractor_doc_ids


def test_generate_instance_visits_all_templates_under_seed_sweep() -> None:
    seen: set[str] = set()
    for seed in range(60):
        seen.add(generate_instance(seed=seed).template_name)
    assert seen == set(TEMPLATE_NAMES)


def test_generate_instance_carries_distinct_chain_and_distractors() -> None:
    """D4-C: chain + distractor docs do not overlap."""
    for seed in range(10):
        inst = generate_instance(seed=seed)
        chain = set(inst.gold_chain_doc_ids)
        distractors = set(inst.distractor_doc_ids)
        assert not (chain & distractors), (
            f"seed {seed}: chain {chain} overlaps distractors {distractors}"
        )
        # Each set has positive size — chain has 2-3, distractors have 1-2.
        assert len(chain) >= 2
        assert len(distractors) >= 1


def test_two_hop_gold_answer_is_numeric() -> None:
    """chain_two_hop returns a population number."""
    for seed in range(60):
        inst = generate_instance(seed=seed)
        if inst.template_name == "chain_two_hop":
            assert inst.gold_answer_kind == "numeric"
            assert isinstance(inst.gold_answer, float)
            return
    pytest.skip("no chain_two_hop instance in 60 seeds")


def test_three_hop_gold_answer_is_string() -> None:
    """chain_three_hop returns a person name."""
    for seed in range(60):
        inst = generate_instance(seed=seed)
        if inst.template_name == "chain_three_hop":
            assert inst.gold_answer_kind == "string"
            assert isinstance(inst.gold_answer, str)
            return
    pytest.skip("no chain_three_hop instance in 60 seeds")


def test_arithmetic_gold_answer_is_numeric() -> None:
    for seed in range(60):
        inst = generate_instance(seed=seed)
        if inst.template_name == "arithmetic_over_facts":
            assert inst.gold_answer_kind == "numeric"
            assert isinstance(inst.gold_answer, float)
            return
    pytest.skip("no arithmetic_over_facts instance in 60 seeds")


def test_as_inputs_excludes_oracle_fields() -> None:
    """D9 — gold chain doc ids are NEVER serialised into the prompt."""
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_answer" not in inputs
    assert "gold_chain_doc_ids" not in inputs
    assert "distractor_doc_ids" not in inputs
    assert "prompt" in inputs
    assert "template_name" in inputs


def test_user_prompt_carries_corpus_and_question() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "QUESTION:" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.question in text


# ── Reward kernel ──────────────────────────────────────────────────


def _toy_prediction(answer: str) -> ReasoningPrediction:
    return ReasoningPrediction(
        answer=answer,
        raw=json.dumps({"answer": answer, "confidence": 0.5}),
        confidence=0.5,
    )


def test_score_components_baseline_zero_everywhere() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0


def test_score_components_garbled_text_short_circuits() -> None:
    inst = generate_instance(seed=0)
    pred = ReasoningPrediction(answer="", raw="not json", confidence=0.0)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0


def test_score_components_correct_numeric_answer() -> None:
    """Submitting the gold numeric value scores correctness=1.0."""
    for seed in range(40):
        inst = generate_instance(seed=seed)
        if inst.gold_answer_kind == "numeric":
            pred = _toy_prediction(str(inst.gold_answer))
            components = score_components(pred, inst)
            assert components["format_valid"] == 1.0
            assert components["correctness"] == pytest.approx(1.0)
            return
    pytest.skip("no numeric instance in seed sweep")


def test_score_components_correct_string_answer() -> None:
    for seed in range(40):
        inst = generate_instance(seed=seed)
        if inst.gold_answer_kind == "string":
            pred = _toy_prediction(str(inst.gold_answer))
            components = score_components(pred, inst)
            assert components["correctness"] == pytest.approx(1.0)
            return
    pytest.skip("no string instance in seed sweep")


def test_score_components_distractor_answer_zero_correctness() -> None:
    """Submitting a distractor's value gets correctness=0."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction("99999.0")  # not the gold answer
    components = score_components(pred, inst)
    if inst.gold_answer_kind == "numeric" and float(inst.gold_answer) != 99999.0:
        assert components["correctness"] == 0.0


# ── Reward aggregation ─────────────────────────────────────────────


def test_compute_reward_in_unit_range_and_emits_meta() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["template"] == inst.template_name
    assert out["meta"]["gold_answer_kind"] == inst.gold_answer_kind
    assert "covered" in out["meta"]


def test_compute_reward_carries_completion_hash_and_cache_key() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction("123.0")
    out = compute_reward(prediction=pred, instance=inst)
    assert "completion_hash" in out["meta"]
    assert "cache_key" in out["meta"]
    assert len(out["meta"]["cache_key"]) == 16


def test_compute_reward_full_credit_for_correct_numeric() -> None:
    for seed in range(40):
        inst = generate_instance(seed=seed)
        if inst.gold_answer_kind == "numeric":
            pred = _toy_prediction(str(inst.gold_answer))
            out = compute_reward(prediction=pred, instance=inst)
            assert out["reward"] == pytest.approx(1.0)
            return
    pytest.skip("no numeric instance in seed sweep")


# ── Adapter ────────────────────────────────────────────────────────


def test_system_prompt_documents_envelope() -> None:
    assert "answer" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    payload = json.dumps({"answer": "FooMayor", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert pred.answer == "FooMayor"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_numeric_answer() -> None:
    """Numeric answers (sent as JSON numbers) are stringified."""
    inst = generate_instance(seed=0)
    payload = json.dumps({"answer": 12345, "confidence": 0.9})
    pred = parse_response(payload, inst)
    assert pred.answer == "12345"


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = "```json\n" + json.dumps({"answer": "X", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.answer == "X"


def test_parse_response_returns_empty_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose", inst)
    assert pred.answer == ""


# ── Env class ──────────────────────────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, LongContextReasoningEnv)
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


def test_env_score_with_correct_answer_full_credit() -> None:
    env = load_environment(calibration_quantile=0.5)
    for seed in range(40):
        inst = env.generate_instance(seed=seed)
        if inst.gold_answer_kind == "numeric":
            pred = _toy_prediction(str(inst.gold_answer))
            out = env.score(pred, inst)
            assert out["reward"] == pytest.approx(1.0)
            return
    pytest.skip("no numeric instance in seed sweep")
