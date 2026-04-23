"""Unit tests for the LLM-solver infrastructure. No network required."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from verifiable_labs_envs.solvers import (
    CompletionResult,
    EnvAdapter,
    FakeLLMSolver,
    LLMSolverError,
    register_adapter,
)

# ---------- CompletionResult ----------


def test_completion_result_is_frozen() -> None:
    import dataclasses

    result = CompletionResult(
        text="hi", prompt_tokens=1, completion_tokens=1, latency_s=0.1, model="fake"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "changed"  # type: ignore[misc]


def test_completion_result_optional_usd_cost() -> None:
    result = CompletionResult(
        text="hi", prompt_tokens=1, completion_tokens=1, latency_s=0.1, model="fake"
    )
    assert result.usd_cost is None


# ---------- FakeLLMSolver ----------


def test_fake_solver_returns_string_verbatim() -> None:
    solver = FakeLLMSolver("hello world")
    out = solver.complete(system="sys", user="usr")
    assert out.text == "hello world"
    assert out.model == "fake"
    assert solver.calls == [("sys", "usr")]


def test_fake_solver_callable_response_receives_system_and_user() -> None:
    solver = FakeLLMSolver(lambda s, u: f"[{s}|{u}]")
    out = solver.complete(system="S", user="U")
    assert out.text == "[S|U]"


def test_fake_solver_list_dequeues_per_call() -> None:
    solver = FakeLLMSolver(["first", "second"])
    assert solver.complete("", "").text == "first"
    assert solver.complete("", "").text == "second"
    with pytest.raises(LLMSolverError, match="empty"):
        solver.complete("", "")


# ---------- Solver.solve() plumbing via an inline adapter ----------


@dataclass
class _ToyInstance:
    payload: int


@dataclass
class _ToyPrediction:
    value: int


class _ToyAdapter(EnvAdapter):
    env_name = "_test_env"
    system_prompt = "You are a test adapter."

    def build_user_prompt(self, instance: _ToyInstance) -> str:
        return f"payload={instance.payload}"

    def parse_response(self, text: str, instance: _ToyInstance) -> _ToyPrediction:
        import json

        data = json.loads(text)
        return _ToyPrediction(value=int(data["value"]))


def test_solve_dispatches_to_registered_adapter() -> None:
    register_adapter(_ToyAdapter())
    solver = FakeLLMSolver('{"value": 42}')
    pred = solver.solve("_test_env", _ToyInstance(payload=7))
    assert pred == _ToyPrediction(value=42)
    # Adapter's system_prompt + build_user_prompt must have been used:
    system, user = solver.calls[0]
    assert system == "You are a test adapter."
    assert user == "payload=7"


def test_solve_raises_for_unknown_env() -> None:
    solver = FakeLLMSolver("anything")
    with pytest.raises(LLMSolverError, match="No LLM adapter"):
        solver.solve("definitely-not-registered-env", _ToyInstance(payload=0))


def test_solve_wraps_parse_failures_in_llmsolver_error() -> None:
    register_adapter(_ToyAdapter())
    solver = FakeLLMSolver("not json at all")
    with pytest.raises(LLMSolverError, match="_ToyAdapter"):
        solver.solve("_test_env", _ToyInstance(payload=1))


def test_solve_preserves_llmsolver_error_from_adapter() -> None:
    class _RaisingAdapter(EnvAdapter):
        env_name = "_test_env_raising"
        system_prompt = ""

        def build_user_prompt(self, instance):  # type: ignore[no-untyped-def]
            return ""

        def parse_response(self, text, instance):  # type: ignore[no-untyped-def]
            raise LLMSolverError("schema missing 'foo'")

    register_adapter(_RaisingAdapter())
    solver = FakeLLMSolver("")
    with pytest.raises(LLMSolverError, match="schema missing 'foo'"):
        solver.solve("_test_env_raising", _ToyInstance(payload=0))


# ---------- OpenRouterSolver skips gracefully without a key ----------


def test_openrouter_solver_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force-unset the key AND flip the module flag our guard reads.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "verifiable_labs_envs.solvers.llm_solver.HAS_OPENROUTER_KEY", False, raising=False
    )
    from verifiable_labs_envs.solvers import OpenRouterSolver

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterSolver(model="anthropic/claude-haiku-4.5")


# ---------- Multi-turn (complete_turns) ----------


def test_complete_turns_string_response_echoes() -> None:
    solver = FakeLLMSolver("echoed response")
    result = solver.complete_turns([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ])
    assert result.text == "echoed response"
    assert result.completion_tokens == 2  # "echoed response"
    assert result.prompt_tokens == 2  # S + U


def test_complete_turns_list_queue_drains_per_call() -> None:
    solver = FakeLLMSolver(["first", "second"])
    a = solver.complete_turns([{"role": "user", "content": "x"}])
    b = solver.complete_turns([{"role": "user", "content": "y"}])
    assert a.text == "first"
    assert b.text == "second"
    with pytest.raises(LLMSolverError, match="empty"):
        solver.complete_turns([{"role": "user", "content": "z"}])


def test_complete_turns_records_full_messages_history() -> None:
    solver = FakeLLMSolver(["turn1", "turn2"])
    solver.complete_turns([
        {"role": "system", "content": "You are X"},
        {"role": "user", "content": "q1"},
    ])
    solver.complete_turns([
        {"role": "system", "content": "You are X"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "q2"},
    ])
    history = solver.turn_calls
    assert len(history) == 2
    assert len(history[0]) == 2
    assert len(history[1]) == 4
    assert history[1][2]["role"] == "assistant"
    assert history[1][2]["content"] == "turn1"


def test_complete_and_complete_turns_track_separate_buffers() -> None:
    solver = FakeLLMSolver(["a", "b"])
    solver.complete(system="S", user="U")
    solver.complete_turns([{"role": "user", "content": "q"}])
    assert len(solver.calls) == 1
    assert len(solver.turn_calls) == 1


def test_envadapter_build_followup_turn_raises_by_default() -> None:
    class _SingleTurnAdapter(EnvAdapter):
        env_name = "toy"
        system_prompt = ""

        def build_user_prompt(self, instance):  # type: ignore[no-untyped-def]
            return ""

        def parse_response(self, text, instance):  # type: ignore[no-untyped-def]
            return None

    adapter = _SingleTurnAdapter()
    with pytest.raises(NotImplementedError, match="multi-turn"):
        adapter.build_followup_turn(history=[], last_prediction=None, instance=None)


def test_envadapter_subclass_can_override_build_followup_turn() -> None:
    class _MultiTurnAdapter(EnvAdapter):
        env_name = "toy-mt"
        system_prompt = ""

        def build_user_prompt(self, instance):  # type: ignore[no-untyped-def]
            return "turn0"

        def parse_response(self, text, instance):  # type: ignore[no-untyped-def]
            return None

        def build_followup_turn(self, history, last_prediction, instance):  # type: ignore[no-untyped-def]
            return f"followup #{len(history)}"

    adapter = _MultiTurnAdapter()
    assert adapter.build_followup_turn([{"a": 1}, {"b": 2}], None, None) == "followup #2"
