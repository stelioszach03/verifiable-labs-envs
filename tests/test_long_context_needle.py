"""Tests for the ``long-context-needle`` env (Phase 27.B)."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.envs.long_context_needle import (
    DEFAULT_HYPERPARAMS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    POSITION_MODES,
    SYSTEM_PROMPT,
    LongContextNeedleEnv,
    NeedlePrediction,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    load_environment,
    parse_response,
    score_components,
)
from verifiable_labs_envs.long_context_primitives import DEFAULT_TEST_TOKENS

# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "long-context-needle"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_27_d7_a() -> None:
    """Reward shape (0.10 + 0.20 + 0.70) is locked across Phases 24-27."""
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_default_hyperparams_carry_alpha_and_caps() -> None:
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert DEFAULT_HYPERPARAMS["target_tokens"] == DEFAULT_TEST_TOKENS
    assert DEFAULT_HYPERPARAMS["document_count"] >= 1
    assert DEFAULT_HYPERPARAMS["max_tokens"] >= DEFAULT_HYPERPARAMS["target_tokens"]


def test_position_modes_locked_at_four() -> None:
    assert set(POSITION_MODES) == {"start", "middle", "end", "random"}


# ── Procedural lattice ────────────────────────────────────────────────


def test_generate_instance_is_deterministic_per_seed() -> None:
    a = generate_instance(seed=11)
    b = generate_instance(seed=11)
    assert a.needle_text == b.needle_text
    assert a.position_mode == b.position_mode
    assert a.needle_anchor.document_id == b.needle_anchor.document_id
    assert a.needle_anchor.char_offset == b.needle_anchor.char_offset


def test_generate_instance_varies_with_seed() -> None:
    """Different seeds should yield different needle tokens."""
    seen: set[str] = set()
    for seed in range(10):
        inst = generate_instance(seed=seed)
        seen.add(inst.metadata["needle_token"])
    # 10 random ABCD-1234 tokens almost certainly hit ≥ 8 distinct values.
    assert len(seen) >= 8


def test_generate_instance_carries_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    assert inst.needle_text
    assert inst.metadata["needle_token"]
    assert inst.gold_answer == inst.needle_text
    assert 0 <= inst.needle_anchor.document_id < len(inst.corpus.documents)
    assert inst.position_mode in POSITION_MODES


def test_as_inputs_excludes_oracle_fields() -> None:
    """Solvers must not see the needle text or anchor."""
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "needle_text" not in inputs
    assert "needle_anchor" not in inputs
    assert "gold_answer" not in inputs
    assert "prompt" in inputs
    assert "context_token_count" in inputs


def test_user_prompt_carries_corpus_and_question() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "QUESTION:" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.question in text
    # The needle text itself appears (it has been injected into the corpus
    # body) — the env's whole job is to test whether the model finds it.
    assert inst.needle_text in text


def test_user_prompt_corpus_separator_format_strict() -> None:
    """``---DOCUMENT N: <title>---`` separator is locked across Phase 27."""
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "---DOCUMENT 0:" in text
    # Final document index = document_count - 1.
    final_idx = len(inst.corpus.documents) - 1
    assert f"---DOCUMENT {final_idx}:" in text


# ── Reward kernel — pure short-circuits ──────────────────────────────


def _toy_prediction(answer: str) -> NeedlePrediction:
    return NeedlePrediction(
        answer=answer,
        raw=json.dumps({"answer": answer, "confidence": 0.5}),
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
    pred = NeedlePrediction(answer="", raw="not json", confidence=0.0)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0


def test_score_components_correct_needle_clears_correctness() -> None:
    """Predicting the needle's distinctive token gets full correctness."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.metadata["needle_token"])
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == pytest.approx(1.0)


def test_score_components_full_needle_text_also_clears_correctness() -> None:
    """The full needle sentence (which contains the token) also passes."""
    inst = generate_instance(seed=3)
    pred = _toy_prediction(inst.needle_text)
    components = score_components(pred, inst)
    assert components["correctness"] == pytest.approx(1.0)


def test_score_components_wrong_token_zeroes_correctness() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction("ZZZZ-9999")  # garbage token
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == 0.0


def test_score_components_case_insensitive_correctness() -> None:
    """D3-A: substring + case-insensitive."""
    inst = generate_instance(seed=0)
    needle_lower = inst.metadata["needle_token"].lower()
    pred = _toy_prediction(needle_lower)
    components = score_components(pred, inst)
    assert components["correctness"] == pytest.approx(1.0)


# ── Reward aggregation ─────────────────────────────────────────────


def test_compute_reward_in_unit_range_and_emits_meta() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["position_mode"] == inst.position_mode
    assert "context_token_count" in out["meta"]
    assert out["meta"]["context_token_count"] > 0
    assert "needle_doc_id" in out["meta"]
    assert "covered" in out["meta"]


def test_compute_reward_carries_completion_hash_and_cache_key() -> None:
    """D10-B: completion-hash-derived cache key for per-process LRU."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.metadata["needle_token"])
    out = compute_reward(prediction=pred, instance=inst)
    assert "completion_hash" in out["meta"]
    assert len(out["meta"]["completion_hash"]) == 16
    assert "cache_key" in out["meta"]
    assert len(out["meta"]["cache_key"]) == 16


def test_compute_reward_cache_key_stable_across_calls() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.metadata["needle_token"])
    a = compute_reward(prediction=pred, instance=inst)
    b = compute_reward(prediction=pred, instance=inst)
    assert a["meta"]["cache_key"] == b["meta"]["cache_key"]


def test_compute_reward_cache_key_changes_with_completion() -> None:
    inst = generate_instance(seed=0)
    a = compute_reward(prediction=_toy_prediction("FOO-1234"), instance=inst)
    b = compute_reward(prediction=_toy_prediction("BAR-5678"), instance=inst)
    assert a["meta"]["cache_key"] != b["meta"]["cache_key"]


def test_compute_reward_full_credit_for_correct_needle() -> None:
    inst = generate_instance(seed=4)
    pred = _toy_prediction(inst.metadata["needle_token"])
    out = compute_reward(prediction=pred, instance=inst)
    assert out["reward"] == pytest.approx(1.0)


# ── Adapter ────────────────────────────────────────────────────────


def test_system_prompt_documents_envelope() -> None:
    assert "answer" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    payload = json.dumps({"answer": "ABCD-1234", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert pred.answer == "ABCD-1234"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = "```json\n" + json.dumps({"answer": "WXYZ-0000", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.answer == "WXYZ-0000"


def test_parse_response_returns_empty_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose", inst)
    assert pred.answer == ""
    assert pred.confidence == 0.0


def test_parse_response_clamps_confidence() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response(json.dumps({"answer": "X", "confidence": 5.0}), inst)
    assert pred.confidence == pytest.approx(1.0)
    pred = parse_response(json.dumps({"answer": "X", "confidence": -1.0}), inst)
    assert pred.confidence == pytest.approx(0.0)


# ── Env class ──────────────────────────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, LongContextNeedleEnv)
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


def test_env_score_with_correct_needle_full_credit() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = _toy_prediction(inst.metadata["needle_token"])
    out = env.score(pred, inst)
    assert out["reward"] == pytest.approx(1.0)
