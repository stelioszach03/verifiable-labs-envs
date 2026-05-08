"""Tests for the ``tool-calling-single`` env (Phase 25.B)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verifiable_labs_envs.envs.tool_calling_single import (
    _TEMPLATES,
    ACTION_VALIDITY_WEIGHT,
    DEFAULT_HYPERPARAMS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    STATE_MATCH_WEIGHT,
    SYSTEM_PROMPT,
    ToolCallingPrediction,
    ToolCallingSingleEnv,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    generate_problem,
    load_environment,
    parse_response,
    score_components,
)
from verifiable_labs_envs.tool_primitives import (
    WorkspaceState,
    dispatch_tool,
    init_state,
)

# ── Catalogue / metadata ─────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "tool-calling-single"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_phase_25_d6_a() -> None:
    """D6-A locks 0.10 format + 0.20 parse + 0.70 correctness."""
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["correctness"] == pytest.approx(0.70)


def test_d2c_correctness_split_locked() -> None:
    """D2-C — 0.30 action_validity + 0.70 final_state_match."""
    assert pytest.approx(0.30) == ACTION_VALIDITY_WEIGHT
    assert pytest.approx(0.70) == STATE_MATCH_WEIGHT
    assert pytest.approx(1.0) == ACTION_VALIDITY_WEIGHT + STATE_MATCH_WEIGHT


def test_default_max_tool_calls_matches_plan() -> None:
    assert DEFAULT_MAX_TOOL_CALLS == 30


def test_default_hyperparams_carry_alpha_and_budget() -> None:
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert DEFAULT_HYPERPARAMS["max_tool_calls"] == DEFAULT_MAX_TOOL_CALLS


# ── Procedural lattice ────────────────────────────────────────────────


def test_template_count_matches_plan() -> None:
    """PHASE_25_PLAN.md §9.1 locks 10 single-turn templates."""
    assert len(_TEMPLATES) == 10


def test_generate_problem_is_deterministic() -> None:
    a = generate_problem(seed=42)
    b = generate_problem(seed=42)
    assert a == b


def test_generate_problem_covers_all_templates() -> None:
    seen = {generate_problem(seed=s)["template_name"] for s in range(80)}
    assert len(seen) == 10  # all 10 templates appear in 80 seeds.


def test_generate_instance_carries_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    assert inst.gold_spec
    assert inst.available_tools
    assert inst.template_name


def test_as_inputs_excludes_gold_spec() -> None:
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_spec" not in inputs
    assert "prompt" in inputs
    assert "available_tools" in inputs


def test_prompt_does_not_leak_gold_spec_payload() -> None:
    """Sweep seeds; the rendered user prompt must not contain the
    raw gold_spec dict (R10 carry-over). Some scalar values inevitably
    appear (a target number is referenced in the prompt) — what we
    check is that no template dumps the dict verbatim."""
    for seed in range(30):
        inst = generate_instance(seed=seed)
        rendered = build_user_prompt(inst)
        assert "gold_spec" not in rendered
        assert repr(inst.gold_spec) not in rendered


# ── Trajectory recording ─────────────────────────────────────────────


def _make_prediction(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    final_text: str = '{"answer": 0, "confidence": 0.5}',
    state: WorkspaceState | None = None,
    seed: int = 0,
) -> ToolCallingPrediction:
    return ToolCallingPrediction(
        tool_calls=tuple(tool_calls or []),
        final_text=final_text,
        final_state=state if state is not None else init_state(seed=seed),
        raw=final_text,
        confidence=0.7,
    )


# ── Reward kernel — pure (no rollout) ────────────────────────────────


def test_score_components_baseline_zero_everywhere() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_garbled_final_text_short_circuits() -> None:
    inst = generate_instance(seed=0)
    pred = _make_prediction(final_text="not json", seed=inst.seed)
    components = score_components(pred, inst)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["correctness"] == 0.0


def test_score_components_format_only_no_state_match() -> None:
    """Final JSON parses but no tool calls + no state mutation = no correctness."""
    inst = generate_instance(seed=0)
    pred = _make_prediction(seed=inst.seed)
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == 0.0


def test_score_components_arithmetic_template_full_credit() -> None:
    """Drive the arithmetic_compute template through the calculator + correct submission."""
    # Find the first arithmetic_compute seed.
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "arithmetic_compute")
    inst = generate_instance(seed=seed)
    state = init_state(seed=seed)
    expr = inst.gold_spec["expr"]
    target = inst.gold_spec["target"]
    res = dispatch_tool("calculator", {"expression": expr}, state)
    assert "value" in res
    pred = ToolCallingPrediction(
        tool_calls=({"name": "calculator", "arguments": {"expression": expr}, "result": res},),
        final_text=json.dumps({"answer": target, "confidence": 0.9}),
        final_state=state,
        raw="",
        confidence=0.9,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["correctness"] == pytest.approx(1.0)


def test_action_validity_drops_with_error_calls() -> None:
    """50% error-payload rate → action_validity = 0.5 → correctness scaled."""
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "arithmetic_compute")
    inst = generate_instance(seed=seed)
    state = init_state(seed=seed)
    bad = {"name": "calculator", "arguments": {}, "result": {"error": "missing"}}
    good_res = dispatch_tool("calculator", {"expression": inst.gold_spec["expr"]}, state)
    good = {"name": "calculator", "arguments": {"expression": inst.gold_spec["expr"]}, "result": good_res}
    pred = ToolCallingPrediction(
        tool_calls=(bad, good),
        final_text=json.dumps({"answer": inst.gold_spec["target"], "confidence": 0.5}),
        final_state=state,
        raw="",
        confidence=0.5,
    )
    components = score_components(pred, inst)
    # action_validity = 1 / 2; final_state_match = 1.
    expected_correctness = ACTION_VALIDITY_WEIGHT * 0.5 + STATE_MATCH_WEIGHT * 1.0
    assert components["correctness"] == pytest.approx(expected_correctness)


def test_score_components_invalid_tool_args_zeros_parse_valid() -> None:
    inst = generate_instance(seed=0)
    state = init_state(seed=inst.seed)
    bad_call = {"name": "calculator", "arguments": "not json", "result": {"error": "x"}}
    pred = ToolCallingPrediction(
        tool_calls=(bad_call,),
        final_text=json.dumps({"answer": 0}),
        final_state=state,
        raw="",
        confidence=0.5,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0


def test_compute_reward_in_unit_range_and_emits_meta() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.5)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["meta"]["template"] == inst.template_name
    assert out["meta"]["n_tool_calls"] == 0
    assert "covered" in out["meta"]


# ── Per-template gold_state predicates ───────────────────────────────


def _drive_template(template_name: str) -> tuple[Any, ToolCallingPrediction]:
    """Find a seed for ``template_name`` and drive the rollout deterministically."""
    seed = next(
        s for s in range(500)
        if generate_problem(s)["template_name"] == template_name
    )
    inst = generate_instance(seed=seed)
    state = init_state(seed=seed, initial_files=inst.initial_files)
    spec = inst.gold_spec
    calls: list[dict[str, Any]] = []

    def _record(name: str, args: dict[str, Any]) -> None:
        result = dispatch_tool(name, args, state)
        calls.append({"name": name, "arguments": args, "result": result})

    if template_name == "arithmetic_compute":
        _record("calculator", {"expression": spec["expr"]})
    elif template_name == "search_and_email":
        _record("web_search", {"query": spec["topic"], "top_k": 3})
        _record(
            "send_message",
            {"to": spec["recipient"], "body": f"Summary on {spec['topic']}"},
        )
    elif template_name == "file_concat":
        _record("read_file", {"path": "a.txt"})
        _record("read_file", {"path": "b.txt"})
        _record("write_file", {"path": "merged.txt", "content": spec["expected"]})
    elif template_name == "compute_then_send":
        _record("calculator", {"expression": f"{int(spec['target'])}"})
        _record(
            "send_message",
            {"to": spec["recipient"], "body": f"Result is {spec['answer_digits']}"},
        )
    elif template_name == "multi_search":
        body_lines: list[str] = []
        for topic in spec["topics"]:
            res = dispatch_tool("web_search", {"query": topic}, state)
            calls.append({"name": "web_search", "arguments": {"query": topic}, "result": res})
            top = res["results"][0]["title"] if res.get("results") else topic
            body_lines.append(f"{topic}: {top}")
        body = "\n".join(body_lines)
        _record("write_file", {"path": spec["out_path"], "content": body})
    elif template_name == "read_search_write":
        _record("read_file", {"path": "note.txt"})
        _record("web_search", {"query": spec["topic"]})
        _record(
            "write_file",
            {"path": spec["out_path"], "content": f"Note: {spec['topic']}; refs found"},
        )
    elif template_name == "outbox_audit":
        _record("web_search", {"query": spec["topic"]})
        for r in spec["recipients"]:
            _record("send_message", {"to": r, "body": f"Update on {spec['topic']}"})
    elif template_name == "nested_calculator":
        _record("calculator", {"expression": spec["expr"]})
    elif template_name == "search_dedup":
        _record("web_search", {"query": spec["topic"]})
        _record("web_search", {"query": spec["topic"] + " details"})
        _record(
            "write_file",
            {
                "path": spec["out_path"],
                "content": "Title A\nTitle B\nTitle A\n",
            },
        )
    elif template_name == "compute_chain":
        _record("calculator", {"expression": "2 + 3"})
        _record("calculator", {"expression": f"{int(spec['target'])}"})
    else:
        raise AssertionError(f"unknown template {template_name}")

    pred = ToolCallingPrediction(
        tool_calls=tuple(calls),
        final_text=json.dumps({"answer": "done", "confidence": 0.8}),
        final_state=state,
        raw="",
        confidence=0.8,
    )
    return inst, pred


@pytest.mark.parametrize("template_name", [
    "arithmetic_compute",
    "search_and_email",
    "file_concat",
    "compute_then_send",
    "multi_search",
    "read_search_write",
    "outbox_audit",
    "nested_calculator",
    "search_dedup",
    "compute_chain",
])
def test_gold_trajectory_clears_state_match(template_name: str) -> None:
    inst, pred = _drive_template(template_name)
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    # action_validity should be 1.0 across the gold trajectory (no errors).
    assert components["correctness"] == pytest.approx(1.0)


# ── Env class ────────────────────────────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, ToolCallingSingleEnv)
    assert env.name == NAME
    assert env.max_tool_calls == DEFAULT_MAX_TOOL_CALLS


def test_max_tool_calls_negative_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_calls"):
        ToolCallingSingleEnv(conformal_quantile=0.5, max_tool_calls=-1)


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


def test_system_prompt_documents_envelope_and_workspace() -> None:
    assert "tool" in SYSTEM_PROMPT.lower()
    assert "answer" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_build_user_prompt_lists_tools_and_workspace() -> None:
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert "AVAILABLE TOOLS" in text
    for tool in inst.available_tools:
        assert tool in text


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    text = json.dumps({"answer": 42, "confidence": 0.7})
    pred = parse_response(text, inst)
    assert pred.confidence == pytest.approx(0.7)
    assert pred.tool_calls == ()


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = "```json\n" + json.dumps({"answer": "x", "confidence": 0.4}) + "\n```"
    pred = parse_response(text, inst)
    assert pred.confidence == pytest.approx(0.4)


def test_parse_response_returns_zero_confidence_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("garbage", inst)
    assert pred.confidence == 0.0


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
    env_name = "tool-calling-single"
    system_prompt = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> ToolCallingPrediction:
        return parse_response(text, instance)


def test_run_rollout_calculator_happy_path() -> None:
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "arithmetic_compute")
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingSingleEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["reward"] == pytest.approx(1.0)
    assert out["meta"]["n_tool_calls"] == 1
    assert out["meta"]["state"]["calculator_history"]


def test_run_rollout_records_tool_call_results() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("calculator", {"expression": "2 + 2"})),
        _Completion(text=json.dumps({"answer": 4, "confidence": 0.5})),
    ])
    env = ToolCallingSingleEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    tool_calls = out["meta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "calculator"
    assert tool_calls[0]["result"] == {"value": 4.0}


def test_run_rollout_caps_tool_calls() -> None:
    inst = generate_instance(seed=0)
    # 5 tool replies + 1 final, but budget is 2.
    replies = [
        _Completion(tool_call=_ToolCall("calculator", {"expression": "1 + 1"}))
        for _ in range(5)
    ] + [_Completion(text=json.dumps({"answer": 0, "confidence": 0.0}))]
    solver = _ScriptedSolver(replies)
    env = ToolCallingSingleEnv(conformal_quantile=0.5, max_tool_calls=2)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 2


def test_run_rollout_max_zero_skips_to_final() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([_Completion(text=json.dumps({"answer": 0, "confidence": 0.0}))])
    env = ToolCallingSingleEnv(conformal_quantile=0.5, max_tool_calls=0)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 0


def test_run_rollout_handles_unknown_tool_softly() -> None:
    """Unknown tool name returns an error payload but does NOT terminate."""
    seed = next(s for s in range(200) if generate_problem(s)["template_name"] == "arithmetic_compute")
    inst = generate_instance(seed=seed)
    spec = inst.gold_spec
    solver = _ScriptedSolver([
        _Completion(tool_call=_ToolCall("frobnicate", {})),
        _Completion(tool_call=_ToolCall("calculator", {"expression": spec["expr"]})),
        _Completion(text=json.dumps({"answer": spec["target"], "confidence": 0.9})),
    ])
    env = ToolCallingSingleEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    # Two tool calls recorded — the unknown one + the recovery.
    assert out["meta"]["n_tool_calls"] == 2
    # action_validity = 1/2 → correctness scales accordingly.
    expected = ACTION_VALIDITY_WEIGHT * 0.5 + STATE_MATCH_WEIGHT * 1.0
    assert out["components"]["correctness"] == pytest.approx(expected)
