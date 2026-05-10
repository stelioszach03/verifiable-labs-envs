"""Tests for the ``code-humaneval-multiturn`` env (Phase 24.C).

Mirrors ``tests/test_math_algebra_multiturn.py``: env contract,
turn-penalty math, visible-test feedback, hidden-test secrecy
(R10), rollout machinery.

Registry-level tests (``load_environment(NAME)``) live in 24.F's
``test_registry.py`` update — Phase 24.C ships the env file but
defers registration to 24.F so each commit stays minimal.
"""
from __future__ import annotations

import json
import sys

import pytest

from verifiable_labs_envs.envs.code_humaneval import (
    CodeHumanevalEnv,
    CodeInstance,
    CodePrediction,
    generate_instance,
)
from verifiable_labs_envs.envs.code_humaneval_multiturn import (
    DEFAULT_MAX_TURNS,
    NAME,
    TURN_PENALTY_CAP,
    TURN_PENALTY_PER_EXTRA,
    CodeHumanevalMultiturnEnv,
    load_environment,
    render_feedback_message,
    visible_test_feedback,
)
from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    _unshare_available as _sandbox_capable,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not _sandbox_capable(),
    reason=(
        "code-humaneval / code-mini-repo scoring requires the Linux "
        "sandbox primitive (unshare -rn). GitHub-hosted ubuntu-latest "
        "runners ship the binary but kernel rejects uid_map writes."
    ),
)


# ── Env contract ─────────────────────────────────────────────────────


def test_env_id_is_kebab_case():
    assert NAME == "code-humaneval-multiturn"
    assert "_" not in NAME


def test_load_environment_returns_subclass():
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, CodeHumanevalMultiturnEnv)
    # Multi-turn must subclass the single-turn env (verifier reuse).
    assert isinstance(env, CodeHumanevalEnv)
    assert env.name == NAME
    assert env.max_turns == DEFAULT_MAX_TURNS


def test_max_turns_zero_rejected():
    with pytest.raises(ValueError, match="max_turns"):
        CodeHumanevalMultiturnEnv(conformal_quantile=0.5, max_turns=0)


def test_load_environment_respects_max_turns_kwarg():
    env = load_environment(calibration_quantile=0.5, max_turns=5)
    assert env.max_turns == 5


# ── Verifier reuse ───────────────────────────────────────────────────


def test_score_delegates_to_single_turn_reward():
    """Multi-turn must produce the same per-call reward as single-turn
    when called via ``.score(prediction, instance)`` directly."""
    inst = generate_instance(seed=0)
    pred = CodePrediction(
        code=inst.gold_solution,
        raw=json.dumps({"code": inst.gold_solution, "confidence": 1.0}),
        confidence=1.0,
    )
    single = CodeHumanevalEnv(conformal_quantile=0.5)
    multi = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    s_out = single.score(pred, inst)
    m_out = multi.score(pred, inst)
    assert s_out["reward"] == m_out["reward"]


# ── Turn-penalty arithmetic ──────────────────────────────────────────


def test_turn_penalty_zero_for_single_turn():
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty({"reward": 1.0, "components": {}, "meta": {}}, n_turns=1)
    assert out["reward"] == pytest.approx(1.0)
    assert out["meta"]["turn_penalty"] == 0.0


def test_turn_penalty_scales_per_extra_turn():
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty({"reward": 1.0, "components": {}, "meta": {}}, n_turns=3)
    # 2 extra turns × 0.05 = 0.10 penalty.
    assert out["meta"]["turn_penalty"] == pytest.approx(2 * TURN_PENALTY_PER_EXTRA)
    assert out["reward"] == pytest.approx(1.0 - 2 * TURN_PENALTY_PER_EXTRA)


def test_turn_penalty_caps_at_ten_percent():
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty({"reward": 1.0, "components": {}, "meta": {}}, n_turns=20)
    assert out["meta"]["turn_penalty"] == pytest.approx(TURN_PENALTY_CAP)
    assert out["reward"] == pytest.approx(1.0 - TURN_PENALTY_CAP)


def test_turn_penalty_constants_match_phase_21():
    """D8-C ruling: same constants as math-algebra-multiturn."""
    assert TURN_PENALTY_PER_EXTRA == 0.05
    assert TURN_PENALTY_CAP == 0.10


# ── Visible-test feedback (no oracle leakage, R10) ───────────────────


def test_feedback_format_invalid_short_circuits():
    inst = generate_instance(seed=0)
    pred = CodePrediction(code="", raw="not json", confidence=0.0)
    fb = visible_test_feedback(pred, inst, timeout_s=2.0)
    assert fb["format_valid"] == 0.0
    assert fb["passed"] == 0


def test_feedback_uncompileable_short_circuits():
    inst = generate_instance(seed=0)
    pred = CodePrediction(
        code="def x(:\n",
        raw=json.dumps({"code": "def x(:\n"}),
        confidence=0.5,
    )
    fb = visible_test_feedback(pred, inst, timeout_s=2.0)
    assert fb["format_valid"] == 1.0
    assert fb["parse_valid"] == 0.0


def test_feedback_gold_solution_passes_visible():
    inst = generate_instance(seed=0)
    pred = CodePrediction(
        code=inst.gold_solution,
        raw=json.dumps({"code": inst.gold_solution}),
        confidence=1.0,
    )
    fb = visible_test_feedback(pred, inst, timeout_s=10.0)
    assert fb["passed"] == fb["total"]
    assert fb["failed"] == 0


def test_feedback_does_not_leak_hidden_tests():
    """The feedback dict must NEVER contain hidden test source.

    R10 — the model gets pass/fail counts on visible only; hidden
    cases are the held-out grading signal."""
    inst = generate_instance(seed=0)
    pred = CodePrediction(
        code="def f(*a, **k): return None",
        raw=json.dumps({"code": "def f(*a, **k): return None"}),
        confidence=0.5,
    )
    fb = visible_test_feedback(pred, inst, timeout_s=10.0)
    rendered = render_feedback_message(fb)
    for hidden in inst.hidden_tests:
        assert hidden not in rendered, "hidden test leaked into feedback"
    # The total reported should equal visible_tests count, NOT
    # visible + hidden.
    assert fb["total"] == len(inst.visible_tests)


def test_render_feedback_message_includes_pass_count():
    fb = {
        "format_valid": 1.0,
        "parse_valid": 1.0,
        "passed": 1,
        "failed": 1,
        "total": 2,
        "error_excerpt": "FAILED test_case_0 - AssertionError",
    }
    msg = render_feedback_message(fb)
    assert "1/2" in msg
    assert "AssertionError" in msg


def test_render_feedback_message_format_invalid():
    fb = {
        "format_valid": 0.0, "parse_valid": 0.0,
        "passed": 0, "failed": 0, "total": 5, "error_excerpt": "",
    }
    msg = render_feedback_message(fb)
    assert "JSON" in msg


def test_render_feedback_message_parse_invalid():
    fb = {
        "format_valid": 1.0, "parse_valid": 0.0,
        "passed": 0, "failed": 0, "total": 5, "error_excerpt": "",
    }
    msg = render_feedback_message(fb)
    assert "compile" in msg or "syntax" in msg.lower()


def test_render_feedback_message_full_visible_pass():
    fb = {
        "format_valid": 1.0, "parse_valid": 1.0,
        "passed": 3, "failed": 0, "total": 3, "error_excerpt": "",
    }
    msg = render_feedback_message(fb)
    assert "3" in msg
    assert "hidden" in msg.lower() or "edge" in msg.lower()


# ── Rollout machinery (mocked LLM) ───────────────────────────────────


class _ScriptedSolver:
    """Minimal LLMSolver-shaped stub that replays canned completions."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.history_log: list[list[dict[str, str]]] = []

    def complete_turns(self, history: list[dict[str, str]]) -> Any:  # noqa: ANN401
        self.history_log.append([dict(m) for m in history])
        if not self._replies:
            raise RuntimeError("solver ran out of canned replies")
        text = self._replies.pop(0)

        class _Completion:
            def __init__(self, t: str) -> None:
                self.text = t

        return _Completion(text)


class _ScriptedAdapter:
    """Minimal EnvAdapter-shaped stub for run_rollout testing."""

    env_name = "code-humaneval-multiturn"
    system_prompt = "test-system"

    def build_user_prompt(self, instance: CodeInstance) -> str:
        return f"prompt for seed {instance.seed}"

    def parse_response(self, text: str, instance: CodeInstance) -> CodePrediction:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return CodePrediction(code="", raw=text, confidence=0.0)
        return CodePrediction(
            code=str(data.get("code", "")),
            raw=text,
            confidence=float(data.get("confidence", 0.0)),
        )

    def build_followup_turn(
        self,
        history: list[dict[str, str]],
        last_prediction: CodePrediction,
        instance: CodeInstance,
    ) -> str:
        del history
        fb = visible_test_feedback(last_prediction, instance)
        return render_feedback_message(fb)


from typing import Any  # noqa: E402  (placed below the test stubs for grouping)


def test_run_rollout_returns_canonical_dict():
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    solver = _ScriptedSolver(
        [
            json.dumps({"code": "def x(): return 0", "confidence": 0.3}),
            json.dumps({"code": "def x(): return 0", "confidence": 0.4}),
            json.dumps({"code": gold, "confidence": 0.9}),
        ]
    )
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert out["meta"]["n_turns"] == 3
    assert len(out["meta"]["turn_rewards"]) == 3


def test_run_rollout_applies_turn_penalty():
    """Final reward is base × (1 − 0.10) for a 3-turn rollout."""
    inst = generate_instance(seed=0)
    gold = inst.gold_solution
    solver = _ScriptedSolver(
        [
            json.dumps({"code": gold, "confidence": 0.9}),
            json.dumps({"code": gold, "confidence": 0.9}),
            json.dumps({"code": gold, "confidence": 0.9}),
        ]
    )
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    base = out["meta"]["base_reward"]
    assert base == pytest.approx(1.0)
    # 2 extra turns × 0.05 = 0.10
    assert out["reward"] == pytest.approx(0.9, abs=0.01)


def test_run_rollout_history_grows_per_turn():
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver(
        [
            json.dumps({"code": "def x(): return None", "confidence": 0.1}),
            json.dumps({"code": inst.gold_solution, "confidence": 0.8}),
            json.dumps({"code": inst.gold_solution, "confidence": 0.9}),
        ]
    )
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5, max_turns=3)
    env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    # 3 calls to complete_turns; the third call's history must contain
    # the assistant + feedback messages from turns 1 and 2.
    third_call_history = solver.history_log[-1]
    roles = [m["role"] for m in third_call_history]
    assert roles.count("assistant") == 2
    assert roles.count("user") == 3  # initial prompt + 2 feedback turns


def test_run_rollout_respects_max_turns_override():
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver(
        [
            json.dumps({"code": inst.gold_solution, "confidence": 0.9}),
            json.dumps({"code": inst.gold_solution, "confidence": 0.9}),
        ]
    )
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter(), max_turns=2)
    assert out["meta"]["n_turns"] == 2
    assert out["meta"]["max_turns"] == 2


def test_run_rollout_records_per_turn_components():
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver(
        [
            json.dumps({"code": "def x(): return None", "confidence": 0.1}),
            json.dumps({"code": inst.gold_solution, "confidence": 0.9}),
            json.dumps({"code": inst.gold_solution, "confidence": 0.9}),
        ]
    )
    env = CodeHumanevalMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    components = out["meta"]["turn_components"]
    assert len(components) == 3
    for c in components:
        for k in ("format_valid", "parse_valid", "pass_rate"):
            assert k in c
            assert 0.0 <= c[k] <= 1.0
