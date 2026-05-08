"""Tests for the ``long-context-synthesis`` env (Phase 27.C)."""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs.envs.long_context_synthesis import (
    DEFAULT_HYPERPARAMS,
    DEFAULT_MAX_TURNS,
    DEFAULT_NEEDLE_COUNT_RANGE,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    TURN_PENALTY_CAP,
    TURN_PENALTY_PER_EXTRA,
    LongContextSynthesisEnv,
    SynthesisInstance,
    SynthesisPrediction,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    load_environment,
    parse_response,
    render_synthesis_feedback,
    score_components,
)

# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "long-context-synthesis"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_27_d7_a() -> None:
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_turn_penalty_constants_match_d6_a_lock() -> None:
    """Phase 27 D6-A: same constants as code/math multiturn families."""
    assert pytest.approx(0.05) == TURN_PENALTY_PER_EXTRA
    assert pytest.approx(0.10) == TURN_PENALTY_CAP
    assert DEFAULT_MAX_TURNS == 3


def test_default_needle_count_range() -> None:
    """D4-B: 3-5 needles per instance."""
    assert DEFAULT_NEEDLE_COUNT_RANGE == (3, 5)
    n_min, n_max = DEFAULT_HYPERPARAMS["needle_count_range"]
    assert n_min == 3
    assert n_max == 5


# ── Procedural lattice ────────────────────────────────────────────────


def test_generate_instance_is_deterministic_per_seed() -> None:
    a = generate_instance(seed=11)
    b = generate_instance(seed=11)
    assert a.needle_facts == b.needle_facts
    assert a.gold_answer == b.gold_answer
    assert a.needle_doc_ids == b.needle_doc_ids


def test_generate_instance_carries_3_to_5_needles() -> None:
    """Sweep seeds: every instance has 3-5 needles in distinct docs."""
    for seed in range(15):
        inst = generate_instance(seed=seed)
        assert 3 <= len(inst.needle_facts) <= 5, (
            f"seed {seed}: needle_count={len(inst.needle_facts)}"
        )
        # Each needle landed in a distinct document.
        assert len(set(inst.needle_doc_ids)) == len(inst.needle_anchors)


def test_generate_instance_gold_answer_includes_each_needle_token() -> None:
    """Gold answer tokens must be present in their respective needles."""
    inst = generate_instance(seed=0)
    # Each needle has a unique XXXX-#### token; the gold answer
    # carries each token verbatim.
    for needle in inst.needle_facts:
        # Find the token in the needle (matches "ABCD-1234").
        import re
        m = re.search(r"[A-Z]{4}-\d{4}", needle)
        assert m, f"needle missing token: {needle}"
        token = m.group(0)
        assert token in inst.gold_answer


def test_as_inputs_excludes_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "needle_facts" not in inputs
    assert "needle_anchors" not in inputs
    assert "gold_answer" not in inputs
    assert "needle_doc_ids" not in inputs
    assert "prompt" in inputs
    assert "needle_count" in inputs


def test_user_prompt_carries_corpus_and_question() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "QUESTION:" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.question in text
    # Each needle should appear in the rendered corpus body.
    for needle in inst.needle_facts:
        assert needle in text


# ── Reward kernel — token-F1 ─────────────────────────────────────────


def _toy_prediction(answer: str) -> SynthesisPrediction:
    return SynthesisPrediction(
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
    pred = SynthesisPrediction(answer="", raw="not json", confidence=0.0)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0


def test_score_components_gold_answer_full_credit() -> None:
    """Submitting the gold answer scores token-F1 ≈ 1.0."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_answer)
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == pytest.approx(1.0)


def test_score_components_partial_credit_is_continuous() -> None:
    """token-F1 returns a graded float in [0, 1] (D7-A continuous)."""
    inst = generate_instance(seed=0)
    # Take only half the gold facts.
    needle_tokens = inst.gold_answer.split(";")
    half_answer = ";".join(needle_tokens[: len(needle_tokens) // 2 + 1])
    pred = _toy_prediction(half_answer)
    components = score_components(pred, inst)
    assert 0.0 < components["correctness"] < 1.0


def test_score_components_unrelated_answer_zero_correctness() -> None:
    """Wrong answer → token-F1 close to 0."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction("the quick brown fox jumps over the lazy dog")
    components = score_components(pred, inst)
    assert components["correctness"] < 0.2


# ── Reward aggregation ─────────────────────────────────────────────


def test_compute_reward_in_unit_range_and_emits_meta() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["needle_count"] == len(inst.needle_facts)
    assert out["meta"]["needle_doc_ids"] == list(inst.needle_doc_ids)
    assert "f1" in out["meta"]
    assert "covered" in out["meta"]


def test_compute_reward_carries_completion_hash_and_cache_key() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_answer)
    out = compute_reward(prediction=pred, instance=inst)
    assert "completion_hash" in out["meta"]
    assert "cache_key" in out["meta"]
    assert len(out["meta"]["cache_key"]) == 16


# ── Adapter ────────────────────────────────────────────────────────


def test_system_prompt_documents_envelope() -> None:
    assert "answer" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    payload = json.dumps({"answer": "the production figure ABCD-1234", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert "ABCD-1234" in pred.answer
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = "```json\n" + json.dumps({"answer": "done", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.answer == "done"


def test_parse_response_returns_empty_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose", inst)
    assert pred.answer == ""


# ── Feedback rendering ─────────────────────────────────────────────


def test_feedback_renderer_three_branches() -> None:
    low = render_synthesis_feedback(f1_score=0.10, needle_doc_ids=(0, 3, 5))
    mid = render_synthesis_feedback(f1_score=0.70, needle_doc_ids=(0, 3, 5))
    high = render_synthesis_feedback(f1_score=0.95, needle_doc_ids=(0, 3, 5))

    assert "FEEDBACK" in low
    assert "10%" in low
    assert "[0, 3, 5]" in low or "0, 3, 5" in low
    assert "FEEDBACK" in mid
    assert "70%" in mid
    assert "FEEDBACK" in high
    assert "largely correct" in high


def test_feedback_does_not_leak_gold_answer() -> None:
    """R10: feedback must NOT contain the gold answer text."""
    inst = generate_instance(seed=0)
    text = render_synthesis_feedback(
        f1_score=0.30, needle_doc_ids=inst.needle_doc_ids,
    )
    # The full gold answer (with all needle phrases) must not appear.
    assert inst.gold_answer not in text
    # Each individual needle should also not be reproduced.
    for needle in inst.needle_facts:
        assert needle not in text


# ── Env class + multi-turn rollout ────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, LongContextSynthesisEnv)
    assert env.name == NAME
    assert env.max_turns == DEFAULT_MAX_TURNS


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


def test_env_score_with_gold_answer_full_credit() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_answer)
    out = env.score(pred, inst)
    assert out["reward"] == pytest.approx(1.0)


def test_apply_turn_penalty_three_turns_yields_0_9() -> None:
    """3-turn rollout → 0.05 × 2 = 0.10 → 0.9× the base reward."""
    env = load_environment(calibration_quantile=0.5)
    scored = {"reward": 1.0, "meta": {}, "components": {}}
    penalised = env._apply_turn_penalty(scored, n_turns=3)
    assert penalised["reward"] == pytest.approx(0.9)
    assert penalised["meta"]["base_reward"] == pytest.approx(1.0)
    assert penalised["meta"]["turn_penalty"] == pytest.approx(0.10)


def test_apply_turn_penalty_one_turn_no_penalty() -> None:
    env = load_environment(calibration_quantile=0.5)
    scored = {"reward": 0.7, "meta": {}, "components": {}}
    penalised = env._apply_turn_penalty(scored, n_turns=1)
    assert penalised["reward"] == pytest.approx(0.7)
    assert penalised["meta"]["turn_penalty"] == pytest.approx(0.0)


def test_apply_turn_penalty_caps_at_ten_percent() -> None:
    env = load_environment(calibration_quantile=0.5)
    scored = {"reward": 1.0, "meta": {}, "components": {}}
    penalised = env._apply_turn_penalty(scored, n_turns=10)
    # 0.05 × 9 = 0.45 → capped at 0.10.
    assert penalised["meta"]["turn_penalty"] == pytest.approx(0.10)
    assert penalised["reward"] == pytest.approx(0.9)


def test_build_followup_turn_uses_f1_branch() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_answer)  # F1 ≈ 1.0 → high branch
    text = env.build_followup_turn(pred, inst)
    assert "largely correct" in text


def test_build_followup_turn_does_not_leak_gold() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = SynthesisPrediction(answer="", raw="", confidence=0.0)
    text = env.build_followup_turn(pred, inst)
    # R10: gold answer text MUST NOT appear in the inter-turn feedback.
    assert inst.gold_answer not in text


def test_run_rollout_applies_turn_penalty() -> None:
    """Three-turn rollout with the gold answer scores 0.9× base."""
    env = load_environment(calibration_quantile=0.5)

    class _GoldSolver:
        """Returns the gold answer text on every turn."""

        def __init__(self, instance: SynthesisInstance) -> None:
            self.instance = instance

        def complete_turns(self, history):  # noqa: ARG002
            payload = {"answer": self.instance.gold_answer, "confidence": 1.0}

            class _C:
                text = json.dumps(payload)

            return _C()

    class _Adapter:
        system_prompt = SYSTEM_PROMPT

        @staticmethod
        def build_user_prompt(instance):
            return build_user_prompt(instance)

        @staticmethod
        def parse_response(text, instance):
            return parse_response(text, instance)

    inst = env.generate_instance(seed=0)
    solver = _GoldSolver(inst)
    out = env.run_rollout(solver, inst, adapter=_Adapter, max_turns=3)
    assert out["reward"] == pytest.approx(0.9)
    assert out["meta"]["n_turns"] == 3
    assert out["meta"]["max_turns"] == 3
    assert out["meta"]["turn_penalty"] == pytest.approx(0.10)
    assert len(out["meta"]["turn_rewards"]) == 3
