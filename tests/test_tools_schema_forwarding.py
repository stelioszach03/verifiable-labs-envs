"""Tests for the validation-fix tools=[...] forwarding plumbing.

Covers:
- ``EnvAdapter.get_tools_schema`` default-None for non-tool envs
- The 5 tool-style adapters override ``get_tools_schema`` and
  return a non-empty OpenAI-format schema list
- ``OpenAICompatibleAgent.solve`` forwards ``tools_schema`` to the
  ``client.chat.completions.create`` call when supplied
- The CLI ``_call_agent`` helper falls through cleanly when the
  agent does not accept ``tools_schema``
"""
from __future__ import annotations

from typing import Any

import pytest

# Importing the adapters package registers every per-env adapter.
import verifiable_labs_envs.solvers.adapters  # noqa: F401
from verifiable_labs_envs import load_environment
from verifiable_labs_envs.agents import OpenAICompatibleAgent
from verifiable_labs_envs.solvers.adapters.code_humaneval_tools import (
    CodeHumanevalToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.math_algebra_tools import (
    MathAlgebraToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.tool_calling_debug import (
    ToolCallingDebugAdapter,
)
from verifiable_labs_envs.solvers.adapters.tool_calling_multiturn import (
    ToolCallingMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.tool_calling_single import (
    ToolCallingSingleAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, get_adapter

# ── EnvAdapter base contract ───────────────────────────────────────


def test_envadapter_default_get_tools_schema_returns_none() -> None:
    """Any adapter that doesn't override should return None."""

    class _Bare(EnvAdapter):
        env_name = "test-bare"
        system_prompt = ""

        def build_user_prompt(self, instance: Any) -> str:
            return ""

        def parse_response(self, text: str, instance: Any) -> Any:
            return None

    assert _Bare().get_tools_schema(instance=None) is None


def test_existing_non_tool_adapters_still_return_none() -> None:
    """Spot-check: non-tool envs must NOT regress to forwarding a schema."""
    for env_id in (
        "math-algebra",
        "code-humaneval",
        "long-context-needle",
        "sql-single-turn",
    ):
        adapter = get_adapter(env_id)
        env = load_environment(env_id)
        instance = env.generate_instance(seed=0)
        assert adapter.get_tools_schema(instance) is None, (
            f"{env_id} unexpectedly forwarded a tools_schema"
        )


# ── Per-tool-adapter schema forwarding ─────────────────────────────


def _assert_openai_schema_shape(schema: list[dict[str, Any]]) -> None:
    """Each entry must be an OpenAI tool-spec object."""
    assert isinstance(schema, list) and schema
    for s in schema:
        assert isinstance(s, dict)
        assert s.get("type") == "function", f"unexpected type: {s}"
        fn = s.get("function") or {}
        assert isinstance(fn, dict)
        assert isinstance(fn.get("name"), str) and fn["name"]
        assert isinstance(fn.get("parameters"), dict)


def test_tool_calling_single_adapter_forwards_schema() -> None:
    env = load_environment("tool-calling-single")
    inst = env.generate_instance(seed=0)
    schema = ToolCallingSingleAdapter().get_tools_schema(inst)
    _assert_openai_schema_shape(schema)


def test_tool_calling_multiturn_adapter_forwards_schema() -> None:
    env = load_environment("tool-calling-multiturn")
    inst = env.generate_instance(seed=0)
    schema = ToolCallingMultiturnAdapter().get_tools_schema(inst)
    _assert_openai_schema_shape(schema)


def test_tool_calling_debug_adapter_forwards_schema() -> None:
    env = load_environment("tool-calling-debug")
    inst = env.generate_instance(seed=0)
    schema = ToolCallingDebugAdapter().get_tools_schema(inst)
    _assert_openai_schema_shape(schema)


def test_math_algebra_tools_adapter_forwards_schema() -> None:
    env = load_environment("math-algebra-tools")
    inst = env.generate_instance(seed=0)
    schema = MathAlgebraToolsAdapter().get_tools_schema(inst)
    _assert_openai_schema_shape(schema)
    names = {s["function"]["name"] for s in schema}
    # Phase 21 D-ruling locks 4 SymPy primitives.
    assert "sympy_simplify" in names
    assert "sympy_expand" in names


def test_code_humaneval_tools_adapter_forwards_schema() -> None:
    env = load_environment("code-humaneval-tools")
    inst = env.generate_instance(seed=0)
    schema = CodeHumanevalToolsAdapter().get_tools_schema(inst)
    _assert_openai_schema_shape(schema)
    names = {s["function"]["name"] for s in schema}
    assert {"read_file", "write_file", "run_test"} <= names


def test_tool_calling_single_per_instance_filtering() -> None:
    """When ``available_tools`` is restricted, the schema list must shrink."""
    env = load_environment("tool-calling-single")
    inst = env.generate_instance(seed=0)
    full = ToolCallingSingleAdapter().get_tools_schema(inst)
    # Synthesise a restricted instance — most adapters look at
    # ``instance.available_tools`` directly.
    class _Stub:
        available_tools = ("calculator",)

    restricted = ToolCallingSingleAdapter().get_tools_schema(_Stub())
    assert len(restricted) < len(full)
    assert {s["function"]["name"] for s in restricted} == {"calculator"}


# ── OpenAICompatibleAgent forwards tools_schema ───────────────────


def _fake_completion(content: str, tool_calls=None):
    """Minimal stand-in for the OpenAI SDK ChatCompletion shape."""

    class _Func:
        def __init__(self, name, args):
            self.name = name
            self.arguments = args

    class _ToolCall:
        def __init__(self, tc_id, fn):
            self.id = tc_id
            self.type = "function"
            self.function = fn

    class _Message:
        def __init__(self):
            self.content = content
            self.tool_calls = (
                [
                    _ToolCall(tc.get("id", "id"),
                             _Func(tc["function"]["name"],
                                   tc["function"]["arguments"]))
                    for tc in tool_calls
                ]
                if tool_calls else None
            )

    class _Choice:
        def __init__(self):
            self.message = _Message()

    class _Resp:
        choices = [_Choice()]

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
        usage = _Usage()

    return _Resp()


class _FakeOpenAIClient:
    """Capture-only fake; records the kwargs the agent forwarded."""

    def __init__(self, response):
        self._response = response
        self.last_kwargs: dict[str, Any] = {}

    class _Chat:
        def __init__(self, parent):
            self.parent = parent

        class _Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, **kwargs):
                self.parent.last_kwargs = kwargs
                return self.parent._response

        @property
        def completions(self):
            return _FakeOpenAIClient._Chat._Completions(self.parent)

    @property
    def chat(self):
        return _FakeOpenAIClient._Chat(self)


def test_openai_agent_forwards_tools_schema(monkeypatch) -> None:
    fake = _FakeOpenAIClient(_fake_completion(content='{"foo":1}'))

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        chat = None  # filled below

    def _factory(**kwargs):  # noqa: ARG001
        return fake


    # Patch the lazy openai import the agent does inside solve().
    class _FakeOpenAIModule:
        OpenAI = _factory

    monkeypatch.setitem(__import__("sys").modules, "openai", _FakeOpenAIModule)

    agent = OpenAICompatibleAgent(
        name="openai:fake",
        model="fake",
        base_url="http://fake.test/v1",
        api_key="sk-fake",
    )
    schema = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object"},
            },
        }
    ]
    out = agent.solve({"prompt_text": "hello"}, tools_schema=schema)
    assert "tools" in fake.last_kwargs
    assert fake.last_kwargs["tools"] == schema
    assert fake.last_kwargs.get("tool_choice") == "auto"
    # response_format must NOT be forwarded when tools= is set.
    assert "response_format" not in fake.last_kwargs
    # The text body still round-trips.
    assert out.get("foo") == 1


def test_openai_agent_default_path_no_tools_kwarg(monkeypatch) -> None:
    """Without tools_schema the agent must NOT forward tools=."""
    fake = _FakeOpenAIClient(_fake_completion(content='{"foo":2}'))

    def _factory(**kwargs):  # noqa: ARG001
        return fake

    class _FakeOpenAIModule:
        OpenAI = _factory

    monkeypatch.setitem(__import__("sys").modules, "openai", _FakeOpenAIModule)

    agent = OpenAICompatibleAgent(
        name="openai:fake",
        model="fake",
        base_url="http://fake.test/v1",
        api_key="sk-fake",
    )
    out = agent.solve({"prompt_text": "hello", "system_prompt": "json please"})
    assert "tools" not in fake.last_kwargs
    assert fake.last_kwargs.get("tool_choice") is None
    assert out.get("foo") == 2


def test_openai_agent_projects_tool_calls_to_canonical_envelope(
    monkeypatch,
) -> None:
    """Returned tool_calls must surface as a list[dict] for adapter parsing."""
    tcs = [
        {
            "id": "call_1",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"2+2"}',
            },
        }
    ]
    fake = _FakeOpenAIClient(
        _fake_completion(content="", tool_calls=tcs),
    )

    def _factory(**kwargs):  # noqa: ARG001
        return fake

    class _FakeOpenAIModule:
        OpenAI = _factory

    monkeypatch.setitem(__import__("sys").modules, "openai", _FakeOpenAIModule)

    agent = OpenAICompatibleAgent(
        name="openai:fake",
        model="fake",
        base_url="http://fake.test/v1",
        api_key="sk-fake",
    )
    schema = [
        {
            "type": "function",
            "function": {"name": "calculator", "parameters": {"type": "object"}},
        }
    ]
    out = agent.solve({"prompt_text": "hi"}, tools_schema=schema)
    assert isinstance(out.get("tool_calls"), list) and out["tool_calls"]
    tc0 = out["tool_calls"][0]
    assert tc0["function"]["name"] == "calculator"
    assert "answer_text" in out  # back-compat fallback


# ── _call_agent fallthrough for legacy agents ─────────────────────


def test_call_agent_falls_through_when_legacy(monkeypatch) -> None:
    from verifiable_labs_envs.cli import _call_agent

    class _LegacyAgent:
        name = "legacy"

        def solve(self, observation):
            return {"echo": observation}

    schema = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    out = _call_agent(_LegacyAgent(), {"prompt_text": "hi"}, schema)
    assert out["echo"]["prompt_text"] == "hi"


def test_call_agent_passes_through_when_modern() -> None:
    from verifiable_labs_envs.cli import _call_agent

    class _ModernAgent:
        name = "modern"

        def solve(self, observation, *, tools_schema=None):
            return {"received_schema_count": len(tools_schema or [])}

    schema = [
        {"type": "function", "function": {"name": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "parameters": {}}},
    ]
    out = _call_agent(_ModernAgent(), {"prompt_text": "hi"}, schema)
    assert out["received_schema_count"] == 2


def test_call_agent_none_tools_skips_kwarg() -> None:
    from verifiable_labs_envs.cli import _call_agent

    class _ModernAgent:
        name = "modern"

        def solve(self, observation, *, tools_schema=None):
            # If _call_agent passed kwargs unconditionally we'd see
            # tools_schema=None. _call_agent must NOT pass kwargs at
            # all in this branch.
            return {"received_kwarg": tools_schema}

    out = _call_agent(_ModernAgent(), {"prompt_text": "hi"}, None)
    assert out["received_kwarg"] is None  # default value, not passed


def test_call_agent_propagates_real_typeerror() -> None:
    from verifiable_labs_envs.cli import _call_agent

    class _BrokenAgent:
        name = "broken"

        def solve(self, observation, *, tools_schema=None):
            raise TypeError("genuine bug unrelated to tools_schema")

    schema = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    with pytest.raises(TypeError, match="genuine bug"):
        _call_agent(_BrokenAgent(), {"prompt_text": "hi"}, schema)
