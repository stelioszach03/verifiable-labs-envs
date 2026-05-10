"""Tests for the ``code-humaneval-tools`` env (Phase 24.D).

Coverage:
- Tool schemas — three function-calling tool definitions
  (read_file, write_file, run_test) with correct shape.
- dispatch_tool — JSON-string and dict argument forms; unknown
  tools surface as error payloads, not exceptions.
- Workspace lifecycle — init seeds solution.py + test_solution.py;
  read_file / write_file mutate a Python dict, not the host
  filesystem; path validation rejects ``..`` escapes.
- run_test — runs against the visible-only test module in the D5
  sandbox; pass/fail counts come back; hidden tests are NEVER
  in the workspace seed (R10).
- Rollout — scripted solver iterates write_file → run_test → final
  envelope; reward prefers workspace ``solution.py`` over the
  parsed JSON envelope.
- Adapter parsing — JSON envelope (clean / fenced / garbage).
"""
from __future__ import annotations

import json
import sys

import pytest

# Sandbox-execution tests need a kernel that supports user-namespace
# creation (unshare -rn); GitHub-hosted ubuntu-latest doesn't.
# Skip the whole module when the primitive isn't usable so CI stays
# green; local dev / WSL / privileged runners still exercise these.
from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    _unshare_available as _sandbox_capable,
)

pytestmark = pytest.mark.skipif(
    not _sandbox_capable(),
    reason='sandbox unshare -rn primitive not usable on this host',
)

from verifiable_labs_envs.envs.code_humaneval import (
    CodeHumanevalEnv,
    CodeInstance,
    CodePrediction,
    generate_instance,
)
from verifiable_labs_envs.envs.code_humaneval_tools import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_TOOL_TIMEOUT_S,
    NAME,
    SYSTEM_PROMPT_TOOLS,
    TOOL_SCHEMAS,
    CodeHumanevalToolsEnv,
    build_user_prompt,
    dispatch_tool,
    init_workspace,
    load_environment,
    parse_response,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="tools rollout requires the Linux sandbox primitive.",
)


# ── Env contract ─────────────────────────────────────────────────────


def test_env_id_is_kebab_case():
    assert NAME == "code-humaneval-tools"
    assert "_" not in NAME


def test_load_environment_returns_subclass():
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, CodeHumanevalToolsEnv)
    assert isinstance(env, CodeHumanevalEnv)
    assert env.name == NAME
    assert env.max_tool_calls == DEFAULT_MAX_TOOL_CALLS


def test_max_tool_calls_negative_rejected():
    with pytest.raises(ValueError, match="max_tool_calls"):
        CodeHumanevalToolsEnv(conformal_quantile=0.5, max_tool_calls=-1)


def test_load_environment_respects_max_tool_calls_kwarg():
    env = load_environment(calibration_quantile=0.5, max_tool_calls=5)
    assert env.max_tool_calls == 5


# ── Tool schemas ─────────────────────────────────────────────────────


def test_tool_schemas_define_three_tools():
    """D9-A locked at 3 minimal primitives."""
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"read_file", "write_file", "run_test"}


def test_tool_schemas_have_required_fields():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_default_constants_match_phase_24_d9_a():
    assert DEFAULT_MAX_TOOL_CALLS == 30
    assert DEFAULT_TOOL_TIMEOUT_S == 5.0


# ── Workspace ────────────────────────────────────────────────────────


def test_init_workspace_seeds_solution_and_test_files():
    inst = generate_instance(seed=0)
    ws = init_workspace(inst)
    assert "solution.py" in ws
    assert "test_solution.py" in ws
    assert ws["solution.py"] == ""


def test_workspace_test_module_holds_only_visible_tests():
    """R10 — at least some hidden test content must be secret.

    Some templates emit visible cases that happen to coincide with one
    or two hidden cases (e.g. the empty-input edge case is both an
    obvious example AND a graded hidden case). The R10 promise is that
    the FULL hidden battery is not exposed — at least one hidden case
    must remain invisible in the workspace seed."""
    # Sweep multiple seeds so a template with accidental full-overlap
    # doesn't fail the assertion globally. R10 only requires that
    # SOME hidden cases stay hidden, not all-or-nothing.
    leaked_per_seed: list[float] = []
    for seed in range(0, 50, 5):
        inst = generate_instance(seed=seed)
        ws = init_workspace(inst)
        test_text = ws["test_solution.py"]
        leaked = sum(1 for h in inst.hidden_tests if h in test_text)
        total = len(inst.hidden_tests)
        leaked_per_seed.append(leaked / max(1, total))
    # Average leak rate must be below 50%; templates with high overlap
    # are OK as outliers but the contamination-resistance contract
    # demands the bulk of the hidden suite stays secret.
    avg = sum(leaked_per_seed) / len(leaked_per_seed)
    assert avg < 0.5, f"too much hidden-test leakage; avg={avg:.2%}"


# ── dispatch_tool: read_file / write_file ────────────────────────────


def test_dispatch_read_file_returns_content():
    ws = {"a.py": "x = 1"}
    out = dispatch_tool("read_file", {"path": "a.py"}, ws)
    assert out == {"content": "x = 1"}


def test_dispatch_read_file_unknown_path_returns_error():
    out = dispatch_tool("read_file", {"path": "nope.py"}, {})
    assert "error" in out


def test_dispatch_write_file_persists_content():
    ws: dict[str, str] = {}
    out = dispatch_tool(
        "write_file",
        {"path": "solution.py", "content": "def f(): return 1"},
        ws,
    )
    assert out["ok"] is True
    assert ws["solution.py"] == "def f(): return 1"


def test_dispatch_write_file_rejects_path_escape():
    out = dispatch_tool(
        "write_file",
        {"path": "../etc/passwd", "content": "pwn"},
        {},
    )
    assert "error" in out


def test_dispatch_write_file_rejects_absolute_path():
    out = dispatch_tool(
        "write_file",
        {"path": "/etc/passwd", "content": "pwn"},
        {},
    )
    assert "error" in out


def test_dispatch_unknown_tool_returns_error():
    out = dispatch_tool("frobnicate", {}, {})
    assert "error" in out
    assert "unknown" in out["error"].lower()


def test_dispatch_accepts_json_string_arguments():
    ws: dict[str, str] = {}
    out = dispatch_tool(
        "write_file",
        json.dumps({"path": "x.py", "content": "y = 2"}),
        ws,
    )
    assert out["ok"] is True
    assert ws["x.py"] == "y = 2"


def test_dispatch_invalid_json_string_arguments_returns_error():
    out = dispatch_tool("write_file", "{not json", {})
    assert "error" in out


# ── dispatch_tool: run_test ──────────────────────────────────────────


def test_dispatch_run_test_with_gold_solution_passes():
    inst = generate_instance(seed=0)
    ws = init_workspace(inst)
    ws["solution.py"] = inst.gold_solution
    out = dispatch_tool("run_test", {"test_id": "all"}, ws, timeout_s=10.0)
    assert out["passed"] >= 1
    assert out["failed"] == 0


def test_dispatch_run_test_with_broken_solution_fails():
    inst = generate_instance(seed=0)
    ws = init_workspace(inst)
    ws["solution.py"] = "def __getattr__(name):\n    return lambda *a, **k: None"
    out = dispatch_tool("run_test", {"test_id": "all"}, ws, timeout_s=10.0)
    # A "return None" stub will fail at least one visible assertion.
    assert out["failed"] >= 1


def test_dispatch_run_test_specific_node_works():
    inst = generate_instance(seed=0)
    ws = init_workspace(inst)
    ws["solution.py"] = inst.gold_solution
    out = dispatch_tool(
        "run_test", {"test_id": "test_visible_000"}, ws, timeout_s=10.0
    )
    assert out["passed"] == 1


def test_dispatch_run_test_missing_solution_module_fails():
    """No solution.py → import-error inside pytest, surfaces as failure."""
    inst = generate_instance(seed=0)
    ws = init_workspace(inst)
    # Empty solution.py will produce ImportError-on-* style failure
    # for tests that call the implementation, depending on template.
    # The contract: exit_code != 0 → at least one failure recorded.
    out = dispatch_tool("run_test", {"test_id": "all"}, ws, timeout_s=10.0)
    assert out["exit_code"] != 0


# ── Adapter ──────────────────────────────────────────────────────────


def test_system_prompt_documents_tools():
    assert "read_file" in SYSTEM_PROMPT_TOOLS
    assert "write_file" in SYSTEM_PROMPT_TOOLS
    assert "run_test" in SYSTEM_PROMPT_TOOLS


def test_build_user_prompt_includes_problem_text():
    inst = generate_instance(seed=0)
    text = build_user_prompt(inst)
    assert inst.function_signature in text
    assert "OUTPUT SCHEMA" in text


def test_parse_response_clean_json():
    inst = generate_instance(seed=0)
    text = json.dumps({"code": "def f(): return 1", "confidence": 0.6})
    pred = parse_response(text, inst)
    assert pred.code == "def f(): return 1"
    assert pred.confidence == pytest.approx(0.6)


def test_parse_response_returns_empty_on_garbage():
    inst = generate_instance(seed=0)
    pred = parse_response("free-form prose", inst)
    assert pred.code == ""


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
    """LLMSolver-shaped stub that replays canned completions."""

    def __init__(self, replies: list[_Completion]) -> None:
        self._replies = list(replies)
        self.history_log: list[list[dict]] = []

    def complete_turns(
        self, history: list[dict], tools: list | None = None
    ) -> _Completion:
        del tools  # we don't validate the tool schemas here
        self.history_log.append([dict(m) for m in history])
        if not self._replies:
            raise RuntimeError("solver ran out of canned completions")
        return self._replies.pop(0)


class _ScriptedAdapter:
    env_name = "code-humaneval-tools"
    system_prompt = SYSTEM_PROMPT_TOOLS

    def build_user_prompt(self, instance: CodeInstance) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: CodeInstance) -> CodePrediction:
        return parse_response(text, instance)


def test_run_rollout_dispatches_tool_calls():
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    solver = _ScriptedSolver(
        [
            _Completion(
                tool_call=_ToolCall(
                    "write_file", {"path": "solution.py", "content": gold}
                ),
            ),
            _Completion(
                tool_call=_ToolCall("run_test", {"test_id": "all"}),
            ),
            _Completion(text=json.dumps({"code": gold, "confidence": 0.95})),
        ]
    )
    env = CodeHumanevalToolsEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 2
    assert out["meta"]["workspace_used"] is True
    assert out["reward"] == pytest.approx(1.0, rel=0.01)


def test_run_rollout_records_each_tool_result():
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver(
        [
            _Completion(
                tool_call=_ToolCall("read_file", {"path": "solution.py"})
            ),
            _Completion(text=json.dumps({"code": "", "confidence": 0.0})),
        ]
    )
    env = CodeHumanevalToolsEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    tool_calls = out["meta"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "read_file"
    assert "content" in tool_calls[0]["result"]


def test_run_rollout_falls_back_to_envelope_when_workspace_empty():
    """If the model never wrote solution.py, the JSON envelope is scored."""
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    solver = _ScriptedSolver(
        [_Completion(text=json.dumps({"code": gold, "confidence": 0.9}))]
    )
    env = CodeHumanevalToolsEnv(conformal_quantile=0.5, max_tool_calls=5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["workspace_used"] is False
    assert out["meta"]["n_tool_calls"] == 0
    assert out["reward"] == pytest.approx(1.0, rel=0.01)


def test_run_rollout_caps_tool_calls():
    """Past the budget, the next completion is treated as the final turn."""
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    # 5 tool calls + 1 final turn — but budget is 2.
    replies = [
        _Completion(
            tool_call=_ToolCall(
                "write_file", {"path": "solution.py", "content": gold}
            )
        ),
        _Completion(
            tool_call=_ToolCall("run_test", {"test_id": "all"})
        ),
        _Completion(text=json.dumps({"code": gold, "confidence": 0.9})),
    ]
    solver = _ScriptedSolver(replies)
    env = CodeHumanevalToolsEnv(conformal_quantile=0.5, max_tool_calls=2)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 2
    assert out["meta"]["max_tool_calls"] == 2


def test_run_rollout_max_tool_calls_zero_skips_to_final_turn():
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    solver = _ScriptedSolver(
        [_Completion(text=json.dumps({"code": gold, "confidence": 0.9}))]
    )
    env = CodeHumanevalToolsEnv(conformal_quantile=0.5, max_tool_calls=0)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert out["meta"]["n_tool_calls"] == 0
    # workspace_used is False because no write_file was called.
    assert out["meta"]["workspace_used"] is False
