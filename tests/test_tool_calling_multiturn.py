"""Tests for the ``tool-calling-multiturn`` env (Phase 25.C)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verifiable_labs_envs.envs.tool_calling_multiturn import (
    NAME,
    TURN_PENALTY_CAP,
    TURN_PENALTY_PER_EXTRA,
    ToolCallingMultiturnEnv,
    load_environment,
    render_turn_feedback,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    ToolCallingPrediction,
    ToolCallingSingleEnv,
    baseline_predict,
    build_user_prompt,
    generate_instance,
    generate_problem,
    parse_response,
)
from verifiable_labs_envs.tool_primitives import init_state

# ── Env contract ─────────────────────────────────────────────────────


def test_env_id_is_kebab_case() -> None:
    assert NAME == "tool-calling-multiturn"
    assert "_" not in NAME


def test_load_environment_returns_subclass() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, ToolCallingMultiturnEnv)
    # Subclass relationship is the verifier-reuse guarantee.
    assert isinstance(env, ToolCallingSingleEnv)
    assert env.name == NAME


def test_max_tool_calls_negative_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_calls"):
        ToolCallingMultiturnEnv(conformal_quantile=0.5, max_tool_calls=-1)


# ── Turn-penalty arithmetic ──────────────────────────────────────────


def test_turn_penalty_zero_for_single_assistant_turn() -> None:
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_assistant_turns=1,
    )
    assert out["reward"] == pytest.approx(1.0)
    assert out["meta"]["turn_penalty"] == 0.0


def test_turn_penalty_scales_per_extra_turn() -> None:
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_assistant_turns=3,
    )
    # 2 extra turns × 0.05 = 0.10 penalty.
    assert out["meta"]["turn_penalty"] == pytest.approx(2 * TURN_PENALTY_PER_EXTRA)
    assert out["reward"] == pytest.approx(1.0 - 2 * TURN_PENALTY_PER_EXTRA)


def test_turn_penalty_caps_at_ten_percent() -> None:
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_assistant_turns=20,
    )
    assert out["meta"]["turn_penalty"] == pytest.approx(TURN_PENALTY_CAP)
    assert out["reward"] == pytest.approx(1.0 - TURN_PENALTY_CAP)


def test_turn_penalty_constants_match_phase_21() -> None:
    """D8-C parity: same constants as math-algebra-multiturn."""
    assert TURN_PENALTY_PER_EXTRA == 0.05
    assert TURN_PENALTY_CAP == 0.10


def test_turn_penalty_records_metadata() -> None:
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 0.8, "components": {}, "meta": {}},
        n_assistant_turns=2,
    )
    assert out["meta"]["base_reward"] == pytest.approx(0.8)
    assert out["meta"]["n_assistant_turns"] == 2


# ── Verifier feedback (no oracle leakage, R10) ───────────────────────


def test_render_feedback_no_call_nudges_to_envelope() -> None:
    msg = render_turn_feedback(last_call=None, n_tool_calls=0, budget=30)
    assert "JSON envelope" in msg or "answer" in msg


def test_render_feedback_error_includes_message_and_remaining() -> None:
    call = {
        "name": "calculator",
        "arguments": {},
        "result": {"error": "calculator: missing expression"},
    }
    msg = render_turn_feedback(last_call=call, n_tool_calls=1, budget=5)
    assert "calculator" in msg
    assert "missing expression" in msg
    assert "remaining" in msg.lower() or "4" in msg


def test_render_feedback_success_includes_summary() -> None:
    call = {
        "name": "calculator",
        "arguments": {"expression": "2 + 2"},
        "result": {"value": 4.0},
    }
    msg = render_turn_feedback(last_call=call, n_tool_calls=1, budget=10)
    assert "calculator" in msg
    assert "succeeded" in msg


def test_render_feedback_does_not_leak_gold_spec_for_any_seed() -> None:
    """Sanity sweep: render feedback against varied tool-call results;
    the gold_spec dict's repr must not appear in any message body."""
    for seed in range(10):
        inst = generate_instance(seed=seed)
        call = {
            "name": "calculator",
            "arguments": {"expression": "1 + 1"},
            "result": {"value": 2.0},
        }
        msg = render_turn_feedback(last_call=call, n_tool_calls=1, budget=30)
        assert repr(inst.gold_spec) not in msg


# ── Verifier reuse ───────────────────────────────────────────────────


def test_score_delegates_to_single_turn_reward() -> None:
    """``.score(prediction, instance)`` produces the same payload as
    the single-turn env. The penalty is layered on inside
    ``run_rollout`` only — direct ``score`` calls stay parity-compatible."""
    inst = generate_instance(seed=0)
    state = init_state(seed=inst.seed)
    pred = ToolCallingPrediction(
        tool_calls=(),
        final_text=json.dumps({"answer": 0, "confidence": 0.5}),
        final_state=state,
        raw="",
        confidence=0.5,
    )
    s_env = ToolCallingSingleEnv(conformal_quantile=0.5)
    m_env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    assert s_env.score(pred, inst)["reward"] == m_env.score(pred, inst)["reward"]


# ── Rollout machinery (mocked LLM) ───────────────────────────────────


class _ToolCall:
    def __init__(self, name: str, arguments: dict | str) -> None:
        self.name = name
        self.arguments = arguments


class _Completion:
    def __init__(self, text: str = "", tool_call: _ToolCall | None = None) -> None:
        self.text = text
        self.tool_call = tool_call


class _ScriptedSolver:
    def __init__(self, replies: list[_Completion]) -> None:
        self._replies = list(replies)
        self.history_log: list[list[dict]] = []

    def complete_turns(
        self, history: list[dict], tools: list | None = None
    ) -> _Completion:
        del tools
        self.history_log.append([dict(m) for m in history])
        if not self._replies:
            raise RuntimeError("solver ran out of canned completions")
        return self._replies.pop(0)


class _ScriptedAdapter:
    env_name = NAME
    system_prompt = "test-system"

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> ToolCallingPrediction:
        return parse_response(text, instance)


def _arithmetic_seed() -> int:
    return next(s for s in range(200) if generate_problem(s)["template_name"] == "arithmetic_compute")


def test_run_rollout_records_assistant_turn_count() -> None:
    seed = _arithmetic_seed()
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_assistant_turns"] == 2
    # 1 extra assistant turn → penalty = 0.05.
    assert out["meta"]["turn_penalty"] == pytest.approx(0.05)
    assert out["reward"] == pytest.approx(0.95, abs=1e-6)


def test_run_rollout_zero_extra_turns_no_penalty() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(text=json.dumps({"answer": 0, "confidence": 0.0})),
    ])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_assistant_turns"] == 1
    assert out["meta"]["turn_penalty"] == 0.0


def test_run_rollout_caps_penalty_at_ten_percent() -> None:
    """Make the model spam tool calls — penalty ceiling enforced."""
    seed = _arithmetic_seed()
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    # 5 tool calls + 1 final = 6 assistant turns → 5 extra → penalty
    # would be 0.25 but capped at 0.10.
    replies = [
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]}))
        for _ in range(5)
    ] + [_Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9}))]
    solver = _ScriptedSolver(replies)
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["turn_penalty"] == pytest.approx(TURN_PENALTY_CAP)


def test_run_rollout_history_carries_feedback_messages() -> None:
    seed = _arithmetic_seed()
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    # The second `complete_turns` call should see the feedback user
    # message appended after the tool result.
    second_history = solver.history_log[-1]
    user_roles = [m for m in second_history if m["role"] == "user"]
    assert len(user_roles) == 2  # initial prompt + feedback turn.
    feedback = user_roles[-1]["content"]
    assert "FEEDBACK" in feedback


def test_run_rollout_softly_recovers_from_unknown_tool() -> None:
    """An unknown tool produces an error result + feedback nudge but does
    NOT terminate the rollout."""
    seed = _arithmetic_seed()
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("frobnicate", {})),
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 2
    # The first call errored → action_validity = 0.5 → correctness blended.
    assert 0.5 < out["reward"] < 1.0


def test_run_rollout_records_state_serialisation() -> None:
    seed = _arithmetic_seed()
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    state = out["meta"]["state"]
    assert "calculator_history" in state
    assert state["calculator_history"]


def test_run_rollout_caps_tool_calls() -> None:
    inst = generate_instance(seed=0)
    replies = [
        _Completion(tool_call=_ToolCall("calculator", {"expression": "1 + 1"}))
        for _ in range(5)
    ] + [_Completion(text=json.dumps({"answer": 0, "confidence": 0.0}))]
    solver = _ScriptedSolver(replies)
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5, max_tool_calls=2)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 2


def test_run_rollout_baseline_returns_canonical_dict() -> None:
    inst = generate_instance(seed=0)
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5)
    pred = baseline_predict(inst)
    out = env.score(pred, inst)
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert "covered" in out["meta"]


def test_load_environment_respects_max_tool_calls_kwarg() -> None:
    env = load_environment(calibration_quantile=0.5, max_tool_calls=10)
    assert env.max_tool_calls == 10


def test_run_rollout_zero_budget_collapses_to_single_turn() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([_Completion(text=json.dumps({"answer": 0, "confidence": 0.0}))])
    env = ToolCallingMultiturnEnv(conformal_quantile=0.5, max_tool_calls=0)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 0
    assert out["meta"]["n_assistant_turns"] == 1
    assert out["meta"]["turn_penalty"] == 0.0
