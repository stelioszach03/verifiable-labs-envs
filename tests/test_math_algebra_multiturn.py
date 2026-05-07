"""Tests for the math-algebra-multiturn env (Phase 21.D).

Mirror of `tests/test_sparse_fourier_multiturn.py`'s structure: env
contract, turn-penalty math, adapter feedback, registry registration.
"""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs.math_algebra import (
    Instance,
    MathAlgebraEnv,
    Prediction,
    generate_instance,
)
from verifiable_labs_envs.envs.math_algebra_multiturn import (
    DEFAULT_MAX_TURNS,
    NAME,
    TURN_PENALTY_CAP,
    TURN_PENALTY_PER_EXTRA,
    MathAlgebraMultiturnEnv,
)
from verifiable_labs_envs.solvers.adapters.math_algebra_multiturn import (
    SYSTEM_PROMPT_MT,
    MathAlgebraMultiturnAdapter,
)

# ───────────────────────── env contract ─────────────────────────────


def test_env_id_is_kebab_case():
    assert NAME == "math-algebra-multiturn"
    assert "_" not in NAME


def test_env_registered_in_central_registry():
    assert NAME in list_environments()


def test_load_environment_via_registry_returns_subclass():
    env = load_environment(NAME, calibration_quantile=0.5)
    assert isinstance(env, MathAlgebraMultiturnEnv)
    # Multi-turn must subclass the single-turn env (verifier reuse).
    assert isinstance(env, MathAlgebraEnv)
    assert env.name == NAME
    assert env.max_turns == DEFAULT_MAX_TURNS


def test_max_turns_zero_rejected():
    with pytest.raises(ValueError, match="max_turns"):
        MathAlgebraMultiturnEnv(conformal_quantile=0.5, max_turns=0)


def test_load_environment_respects_max_turns_kwarg():
    env = load_environment(NAME, calibration_quantile=0.5, max_turns=5)
    assert env.max_turns == 5


# ───────────────────────── verifier reuse ───────────────────────────


def test_score_delegates_to_single_turn_reward():
    """The multi-turn env must produce the same reward as the
    single-turn env on the same (prediction, instance) pair."""
    inst = generate_instance(seed=3)
    pred = Prediction(
        answer_expr=inst.gold_expr,
        raw=json.dumps({"answer": inst.gold_expr, "confidence": 1.0}),
        confidence=1.0,
    )
    single = MathAlgebraEnv(conformal_quantile=0.5)
    multi = MathAlgebraMultiturnEnv(conformal_quantile=0.5)
    s_out = single.score(pred, inst)
    m_out = multi.score(pred, inst)
    # Single-turn score has no turn penalty applied at this layer.
    assert s_out["reward"] == m_out["reward"]


# ───────────────────────── turn penalty ─────────────────────────────


def test_turn_penalty_constants_within_spec():
    """Spec: ≤ 0.1 of total reward."""
    assert pytest.approx(0.05) == TURN_PENALTY_PER_EXTRA
    assert pytest.approx(0.10) == TURN_PENALTY_CAP


def test_apply_turn_penalty_one_turn_no_change():
    env = MathAlgebraMultiturnEnv(conformal_quantile=0.5)
    scored = {"reward": 1.0, "components": {}, "meta": {}}
    out = env._apply_turn_penalty(scored, n_turns=1)
    assert out["reward"] == pytest.approx(1.0)
    assert out["meta"]["turn_penalty"] == pytest.approx(0.0)
    assert out["meta"]["base_reward"] == pytest.approx(1.0)


def test_apply_turn_penalty_two_turns_five_percent():
    env = MathAlgebraMultiturnEnv(conformal_quantile=0.5)
    scored = {"reward": 1.0, "components": {}, "meta": {}}
    out = env._apply_turn_penalty(scored, n_turns=2)
    assert out["reward"] == pytest.approx(0.95)
    assert out["meta"]["turn_penalty"] == pytest.approx(0.05)


def test_apply_turn_penalty_three_turns_caps_at_ten_percent():
    env = MathAlgebraMultiturnEnv(conformal_quantile=0.5)
    scored = {"reward": 1.0, "components": {}, "meta": {}}
    out = env._apply_turn_penalty(scored, n_turns=3)
    assert out["reward"] == pytest.approx(0.9)
    assert out["meta"]["turn_penalty"] == pytest.approx(0.1)


def test_apply_turn_penalty_four_turns_still_capped():
    """Penalty caps at TURN_PENALTY_CAP regardless of turn count."""
    env = MathAlgebraMultiturnEnv(conformal_quantile=0.5)
    scored = {"reward": 1.0, "components": {}, "meta": {}}
    out = env._apply_turn_penalty(scored, n_turns=10)
    assert out["meta"]["turn_penalty"] == pytest.approx(0.1)
    assert out["reward"] == pytest.approx(0.9)


# ───────────────────────── adapter feedback ─────────────────────────


def test_adapter_env_name_matches_registry_key():
    adapter = MathAlgebraMultiturnAdapter()
    assert adapter.env_name == NAME


def test_system_prompt_describes_three_turn_protocol():
    assert "3 turns" in SYSTEM_PROMPT_MT
    assert "FEEDBACK" in SYSTEM_PROMPT_MT


def test_followup_feedback_tells_correct_when_right():
    inst = Instance(
        prompt="Simplify x", gold_expr="x", seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0, "template": "toy"},
    )
    pred = Prediction(
        answer_expr="x",
        raw=json.dumps({"answer": "x", "confidence": 0.9}),
        confidence=0.9,
    )
    adapter = MathAlgebraMultiturnAdapter()
    feedback = adapter.build_followup_turn([], pred, inst)
    assert "correct" in feedback.lower()
    # Must NOT leak the gold expression.
    # ("x" is too short to test for leakage cleanly — use a longer gold below.)


def test_followup_feedback_does_not_leak_gold():
    inst = Instance(
        prompt="Simplify (x+5)*(x-5)", gold_expr="x**2 - 25", seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0, "template": "toy"},
    )
    pred = Prediction(
        answer_expr="x**2 + 25",  # wrong sign
        raw=json.dumps({"answer": "x**2 + 25", "confidence": 0.5}),
        confidence=0.5,
    )
    adapter = MathAlgebraMultiturnAdapter()
    feedback = adapter.build_followup_turn([], pred, inst)
    # The gold "x**2 - 25" must not appear in the feedback.
    assert "x**2 - 25" not in feedback


def test_followup_feedback_distinguishes_failure_modes():
    inst = Instance(
        prompt="Simplify x", gold_expr="x", seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0, "template": "toy"},
    )
    adapter = MathAlgebraMultiturnAdapter()

    bad_json = Prediction(answer_expr="x", raw="not json", confidence=0.5)
    feedback_json = adapter.build_followup_turn([], bad_json, inst)
    assert "JSON" in feedback_json

    bad_parse = Prediction(
        answer_expr="not sympy at all !",
        raw=json.dumps({"answer": "not sympy at all !", "confidence": 0.5}),
        confidence=0.5,
    )
    feedback_parse = adapter.build_followup_turn([], bad_parse, inst)
    assert "SymPy" in feedback_parse or "syntax" in feedback_parse.lower()


def test_adapter_registered_globally():
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  triggers registration
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert NAME in _ADAPTERS
