"""Tests for the ``tool-calling-debug`` env (Phase 25.D)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verifiable_labs_envs.envs.tool_calling_debug import (
    _TEMPLATES,
    DEBUG_HYPERPARAMS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    ToolCallingDebugEnv,
    baseline_predict,
    build_user_prompt,
    check_gold_state,
    generate_instance,
    generate_problem,
    load_environment,
    parse_response,
)
from verifiable_labs_envs.envs.tool_calling_single import (
    ToolCallingPrediction,
    ToolCallingSingleEnv,
)
from verifiable_labs_envs.tool_primitives import (
    WorkspaceState,
    dispatch_tool,
)

# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "tool-calling-debug"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_template_count_matches_plan() -> None:
    """PHASE_25_PLAN.md §9.3 locks 3 trace-debug templates."""
    assert len(_TEMPLATES) == 3


def test_default_hyperparams_inherit_single_turn_alpha() -> None:
    assert "alpha" in DEBUG_HYPERPARAMS
    assert "max_remaining_calls" in DEBUG_HYPERPARAMS


def test_system_prompt_documents_trace_debug_shape() -> None:
    assert "TRACE-DEBUG" in SYSTEM_PROMPT or "trace" in SYSTEM_PROMPT.lower()


# ── Procedural lattice ────────────────────────────────────────────────


def test_generate_problem_is_deterministic() -> None:
    a = generate_problem(seed=42)
    b = generate_problem(seed=42)
    assert a["template_name"] == b["template_name"]
    assert a["base_template"] == b["base_template"]
    assert a["prompt"] == b["prompt"]
    assert a["gold_spec"] == b["gold_spec"]


def test_generate_problem_covers_all_templates() -> None:
    seen = {generate_problem(seed=s)["template_name"] for s in range(60)}
    assert seen == {"partial_compute", "partial_search", "partial_workspace"}


def test_generate_instance_carries_prefix_messages() -> None:
    inst = generate_instance(seed=0)
    assert inst.prefix_messages
    # Each tool call generates two messages (assistant + tool).
    assert len(inst.prefix_messages) % 2 == 0
    # Budget shrinks by the number of pre-computed calls.
    assert inst.max_remaining_calls < 30


def test_generate_instance_carries_replayed_state() -> None:
    """The prefix_state snapshot must reflect the pre-computed tool calls."""
    inst = generate_instance(seed=0)
    state = inst.prefix_state
    assert isinstance(state, WorkspaceState)
    if inst.template_name == "partial_compute":
        assert state.calculator_history
    elif inst.template_name == "partial_search":
        assert state.web_search_calls
    elif inst.template_name == "partial_workspace":
        # read_file mutates web_search_calls? No — read_file does NOT
        # mutate any list, but the prefix has 2 read_file calls; the
        # state's `files` carry the seeded contents.
        assert state.files


def test_as_inputs_excludes_oracle() -> None:
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_spec" not in inputs
    assert "prompt" in inputs
    assert "prefix_messages" in inputs


def test_prefix_does_not_leak_gold_in_prompt() -> None:
    """The trace-debug prompt must NEVER include the gold dict verbatim."""
    for seed in range(0, 60, 6):
        inst = generate_instance(seed=seed)
        # We only assert: no `gold_spec` dict repr leaks. Some
        # natural-language prompts mention the expression to compute,
        # which is intentional (the model needs to know the goal).
        assert repr(inst.gold_spec) not in inst.prompt


# ── Reward kernel ────────────────────────────────────────────────────


def test_baseline_scores_zero() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.score(pred, inst)
    assert out["reward"] == 0.0


def test_check_gold_state_uses_base_template_predicate() -> None:
    """Drive the prefix to completion; the predicate from the base
    template should fire."""
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_compute")
    inst = generate_instance(seed=seed)
    state = inst.prefix_state
    spec = inst.gold_spec
    # Continue the trace: the prefix already computed (a+b); now multiply by c.
    rest = spec["expr"].split("*")[1].strip()  # e.g. "5"
    # The prefix calculator history's last value is (a+b).
    last_partial = float(state.calculator_history[-1].rsplit("=", 1)[-1].strip())
    final_expr = f"{last_partial} * {rest}"
    res = dispatch_tool("calculator", {"expression": final_expr}, state)
    assert "value" in res
    assert check_gold_state(state, inst)


def test_score_full_trajectory_clears_correctness() -> None:
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_workspace")
    inst = generate_instance(seed=seed)
    state = WorkspaceState.from_serialisable(inst.prefix_state_payload)
    spec = inst.gold_spec
    res = dispatch_tool(
        "write_file",
        {"path": spec["out_path"], "content": spec["expected"]},
        state,
    )
    pred = ToolCallingPrediction(
        tool_calls=({"name": "write_file", "arguments": {"path": spec["out_path"], "content": spec["expected"]}, "result": res},),
        final_text=json.dumps({"answer": "done", "confidence": 0.9}),
        final_state=state,
        raw="",
        confidence=0.9,
    )
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.score(pred, inst)
    assert out["components"]["format_valid"] == 1.0
    assert out["components"]["parse_valid"] == 1.0
    assert out["components"]["correctness"] == pytest.approx(1.0)


# ── Env class ────────────────────────────────────────────────────────


def test_load_environment_returns_subclass() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, ToolCallingDebugEnv)
    # Subclass relationship preserves the verifier-reuse guarantee.
    assert isinstance(env, ToolCallingSingleEnv)
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


# ── Adapter ──────────────────────────────────────────────────────────


def test_build_user_prompt_announces_trace_debug() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "TRACE-DEBUG" in text
    assert str(inst.max_remaining_calls) in text


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    text = json.dumps({"answer": 42, "confidence": 0.8})
    pred = parse_response(text, inst)
    assert pred.confidence == pytest.approx(0.8)


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
    system_prompt = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> ToolCallingPrediction:
        return parse_response(text, instance)


def test_run_rollout_seeds_history_with_prefix() -> None:
    """The first solver call must see the prefix messages in history."""
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(text=json.dumps({"answer": 0, "confidence": 0.0})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    history = solver.history_log[0]
    # Roles: system + initial user + (assistant + tool)*prefix_calls + nudge.
    assistant_count = sum(1 for m in history if m["role"] == "assistant")
    tool_count = sum(1 for m in history if m["role"] == "tool")
    assert assistant_count == len(inst.prefix_messages) // 2
    assert tool_count == len(inst.prefix_messages) // 2


def test_run_rollout_partial_compute_full_credit() -> None:
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_compute")
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    # The prefix already computed (a + b). Pull out (a+b) value from
    # the prefix state, then ask the model to multiply by c.
    last_partial = float(inst.prefix_state.calculator_history[-1].rsplit("=", 1)[-1].strip())
    rest = spec["expr"].split("*")[1].strip()
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": f"{last_partial} * {rest}"})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 1
    assert out["meta"]["prefix_len"] >= 1
    assert out["reward"] == pytest.approx(1.0)


def test_run_rollout_partial_workspace_full_credit() -> None:
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_workspace")
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("write_file", {"path": spec["out_path"], "content": spec["expected"]})),
        _Completion(text=json.dumps({"answer": "merged", "confidence": 0.95})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["reward"] == pytest.approx(1.0)


def test_run_rollout_partial_search_full_credit() -> None:
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_search")
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall(
            "send_message",
            {"to": spec["recipient"], "body": f"Summary on {spec['topic']}: see search results."},
        )),
        _Completion(text=json.dumps({"answer": "sent", "confidence": 0.9})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["reward"] == pytest.approx(1.0)


def test_run_rollout_caps_at_remaining_budget() -> None:
    inst = generate_instance(seed=0)
    # Spam tool calls — should be capped by max_remaining_calls.
    replies = [
        _Completion(tool_call=_ToolCall("calculator", {"expression": "1 + 1"}))
        for _ in range(50)
    ] + [_Completion(text=json.dumps({"answer": 0, "confidence": 0.0}))]
    solver = _ScriptedSolver(replies)
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    # Budget = inst.max_remaining_calls; n_tool_calls must not exceed it.
    assert out["meta"]["n_tool_calls"] <= inst.max_remaining_calls


def test_run_rollout_records_prefix_len_in_meta() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(text=json.dumps({"answer": 0, "confidence": 0.0})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["prefix_len"] == len(inst.prefix_messages) // 2


def test_run_rollout_state_starts_from_prefix_snapshot() -> None:
    """The rollout's initial state must equal `inst.prefix_state`."""
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "partial_compute")
    inst = generate_instance(seed=seed)
    solver = _ScriptedSolver([
        _Completion(text=json.dumps({"answer": 0, "confidence": 0.0})),
    ])
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    state = out["meta"]["state"]
    # Calculator history must include the prefix-computed entry.
    assert state["calculator_history"]
    assert state["calculator_history"][0] == inst.prefix_state.calculator_history[0]


def test_load_environment_respects_max_tool_calls_kwarg() -> None:
    env = load_environment(calibration_quantile=0.5, max_tool_calls=10)
    assert env.max_tool_calls == 10


def test_max_tool_calls_negative_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_calls"):
        ToolCallingDebugEnv(conformal_quantile=0.5, max_tool_calls=-1)


def test_prefix_state_is_fresh_per_instance() -> None:
    """Two instances on the same seed share prefix_state shape but
    create independent dicts (mutating one must not bleed)."""
    a = generate_instance(seed=0)
    b = generate_instance(seed=0)
    sa = a.prefix_state
    sb = b.prefix_state
    # Mutate one — the other is unaffected (frozen dataclass guarantee).
    sa.files["intruder.txt"] = "x"
    assert "intruder.txt" not in sb.files


def test_baseline_predict_uses_prefix_state() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    # The "starting point" workspace is the prefix snapshot.
    assert pred.final_state.to_serialisable() == inst.prefix_state_payload


def test_check_gold_state_helper_matches_env_score() -> None:
    """Top-level helper is consistent with the env's internal predicate."""
    inst = generate_instance(seed=0)
    state = inst.prefix_state
    expected = check_gold_state(state, inst)
    pred = ToolCallingPrediction(
        tool_calls=(),
        final_text=json.dumps({"answer": 0, "confidence": 0.0}),
        final_state=state,
        raw="",
        confidence=0.0,
    )
    env = ToolCallingDebugEnv(conformal_quantile=0.5)
    out = env.score(pred, inst)
    correctness_state_match = (
        out["components"]["correctness"] >= 0.7  # state weight in D2-C
    )
    assert correctness_state_match == expected
