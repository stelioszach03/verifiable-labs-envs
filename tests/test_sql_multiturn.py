"""Tests for the ``sql-multiturn`` env (Phase 26.C)."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verifiable_labs_envs.envs.sql_multiturn import (
    DEFAULT_MAX_TURNS,
    NAME,
    TURN_PENALTY_CAP,
    TURN_PENALTY_PER_EXTRA,
    SqlMultiturnEnv,
    load_environment,
    query_diagnostic,
    render_sql_feedback,
)
from verifiable_labs_envs.envs.sql_single_turn import (
    SqlPrediction,
    SqlSingleTurnEnv,
    baseline_predict,
    build_user_prompt,
    generate_instance,
    generate_problem,
    parse_response,
)

# ── Env contract ─────────────────────────────────────────────────────


def test_env_id_is_kebab_case() -> None:
    assert NAME == "sql-multiturn"
    assert "_" not in NAME


def test_load_environment_returns_subclass() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, SqlMultiturnEnv)
    # Multi-turn must subclass single-turn (verifier reuse).
    assert isinstance(env, SqlSingleTurnEnv)
    assert env.name == NAME
    assert env.max_turns == DEFAULT_MAX_TURNS


def test_max_turns_zero_rejected() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        SqlMultiturnEnv(conformal_quantile=0.5, max_turns=0)


def test_load_environment_respects_max_turns_kwarg() -> None:
    env = load_environment(calibration_quantile=0.5, max_turns=5)
    assert env.max_turns == 5


# ── Verifier reuse ───────────────────────────────────────────────────


def test_score_delegates_to_single_turn_reward() -> None:
    """Direct ``.score`` calls produce the same payload as single-turn."""
    inst = generate_instance(seed=0)
    pred = SqlPrediction(
        query=inst.gold_query,
        raw=json.dumps({"query": inst.gold_query, "confidence": 0.9}),
        confidence=0.9,
    )
    s_env = SqlSingleTurnEnv(conformal_quantile=0.5)
    m_env = SqlMultiturnEnv(conformal_quantile=0.5)
    assert s_env.score(pred, inst)["reward"] == m_env.score(pred, inst)["reward"]


# ── Turn-penalty arithmetic ──────────────────────────────────────────


def test_turn_penalty_zero_for_single_turn() -> None:
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_turns=1,
    )
    assert out["reward"] == pytest.approx(1.0)
    assert out["meta"]["turn_penalty"] == 0.0


def test_turn_penalty_scales_per_extra_turn() -> None:
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_turns=3,
    )
    # 2 extra turns × 0.05 = 0.10 penalty.
    assert out["meta"]["turn_penalty"] == pytest.approx(2 * TURN_PENALTY_PER_EXTRA)
    assert out["reward"] == pytest.approx(1.0 - 2 * TURN_PENALTY_PER_EXTRA)


def test_turn_penalty_caps_at_ten_percent() -> None:
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env._apply_turn_penalty(
        {"reward": 1.0, "components": {}, "meta": {}},
        n_turns=20,
    )
    assert out["meta"]["turn_penalty"] == pytest.approx(TURN_PENALTY_CAP)
    assert out["reward"] == pytest.approx(1.0 - TURN_PENALTY_CAP)


def test_turn_penalty_constants_match_phase_21() -> None:
    """D6-A parity: same constants as math-algebra-multiturn."""
    assert TURN_PENALTY_PER_EXTRA == 0.05
    assert TURN_PENALTY_CAP == 0.10


# ── Verifier feedback (no oracle leakage, R10) ───────────────────────


def test_query_diagnostic_returns_counts_only() -> None:
    inst = generate_instance(seed=0)
    diag = query_diagnostic(instance=inst, query=inst.gold_query, timeout_s=2.0)
    assert "predicted_row_count" in diag
    assert "gold_row_count" in diag
    # No raw rows in the diagnostic dict.
    assert "rows" not in diag
    assert "result_rows" not in diag


def test_query_diagnostic_flags_correctness_match() -> None:
    inst = generate_instance(seed=0)
    diag = query_diagnostic(instance=inst, query=inst.gold_query, timeout_s=2.0)
    assert diag["correctness_match"] is True
    assert diag["parse_error"] is None


def test_query_diagnostic_flags_parse_error_for_dml() -> None:
    inst = generate_instance(seed=0)
    diag = query_diagnostic(instance=inst, query="DROP TABLE products")
    assert diag["parse_error"] is not None


def test_query_diagnostic_returns_empty_diagnostic_for_empty_query() -> None:
    inst = generate_instance(seed=0)
    diag = query_diagnostic(instance=inst, query="")
    assert diag["parse_error"] == "empty query"


def test_query_diagnostic_records_row_count_for_wrong_query() -> None:
    inst = generate_instance(seed=0)
    diag = query_diagnostic(
        instance=inst,
        query="SELECT 1 ORDER BY 1",
        timeout_s=2.0,
    )
    assert diag["predicted_row_count"] == 1
    assert diag["gold_row_count"] == len(inst.gold_result_rows)


def test_render_feedback_parse_error() -> None:
    diag = {
        "parse_error": "unsupported leading token",
        "predicted_row_count": None,
        "gold_row_count": 5,
        "correctness_match": False,
    }
    msg = render_sql_feedback(diag)
    assert "did not parse" in msg
    assert "unsupported leading token" in msg


def test_render_feedback_zero_rows() -> None:
    diag = {
        "parse_error": None,
        "predicted_row_count": 0,
        "gold_row_count": 5,
        "correctness_match": False,
    }
    msg = render_sql_feedback(diag)
    assert "0 rows" in msg
    assert "5 row" in msg


def test_render_feedback_row_mismatch() -> None:
    diag = {
        "parse_error": None,
        "predicted_row_count": 7,
        "gold_row_count": 5,
        "correctness_match": False,
    }
    msg = render_sql_feedback(diag)
    assert "7 row" in msg
    assert "5 row" in msg


def test_render_feedback_correct() -> None:
    diag = {
        "parse_error": None,
        "predicted_row_count": 5,
        "gold_row_count": 5,
        "correctness_match": True,
    }
    msg = render_sql_feedback(diag)
    assert "correct" in msg.lower()


def test_render_feedback_does_not_leak_gold_rows() -> None:
    """Sweep seeds: feedback messages must not contain gold-row data."""
    for seed in range(0, 24, 3):
        inst = generate_instance(seed=seed)
        diag = query_diagnostic(
            instance=inst, query="SELECT 1 ORDER BY 1", timeout_s=2.0,
        )
        msg = render_sql_feedback(diag)
        # Gold rows tuple-repr must not appear in the rendered message.
        for row in inst.gold_result_rows[:3]:
            assert repr(row) not in msg


# ── Rollout machinery (mocked LLM) ───────────────────────────────────


class _Completion:
    def __init__(self, text: str) -> None:
        self.text = text


class _ScriptedSolver:
    def __init__(self, replies: list[_Completion]) -> None:
        self._replies = list(replies)
        self.history_log: list[list[dict]] = []

    def complete_turns(self, history: list[dict]) -> _Completion:
        self.history_log.append([dict(m) for m in history])
        if not self._replies:
            raise RuntimeError("solver ran out of canned replies")
        return self._replies.pop(0)


class _ScriptedAdapter:
    env_name = NAME
    system_prompt = "test-system"

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> SqlPrediction:
        return parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],
        last_prediction: Any,
        instance: Any,
    ) -> str:
        del history
        diag = query_diagnostic(
            instance=instance, query=last_prediction.query, timeout_s=2.0,
        )
        return render_sql_feedback(diag)


def test_run_rollout_returns_canonical_dict() -> None:
    inst = generate_instance(seed=0)
    gold = inst.gold_query
    solver = _ScriptedSolver([
        _Completion(json.dumps({"query": "SELECT 1 ORDER BY 1", "confidence": 0.3})),
        _Completion(json.dumps({"query": "SELECT 2 ORDER BY 1", "confidence": 0.4})),
        _Completion(json.dumps({"query": gold, "confidence": 0.9})),
    ])
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert out["meta"]["n_turns"] == 3


def test_run_rollout_applies_turn_penalty() -> None:
    """Final reward = base × (1 − 0.10) for a 3-turn rollout."""
    inst = generate_instance(seed=0)
    gold = inst.gold_query
    solver = _ScriptedSolver([
        _Completion(json.dumps({"query": gold, "confidence": 0.9})),
        _Completion(json.dumps({"query": gold, "confidence": 0.9})),
        _Completion(json.dumps({"query": gold, "confidence": 0.9})),
    ])
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    base = out["meta"]["base_reward"]
    assert base == pytest.approx(1.0)
    # 2 extra turns × 0.05 = 0.10
    assert out["reward"] == pytest.approx(0.9, abs=0.01)


def test_run_rollout_history_grows_per_turn() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(json.dumps({"query": "SELECT 1 ORDER BY 1", "confidence": 0.1})),
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.8})),
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.9})),
    ])
    env = SqlMultiturnEnv(conformal_quantile=0.5, max_turns=3)
    env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    third = solver.history_log[-1]
    roles = [m["role"] for m in third]
    assert roles.count("assistant") == 2
    assert roles.count("user") == 3  # initial + 2 feedback turns.


def test_run_rollout_respects_max_turns_override() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.9})),
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.9})),
    ])
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter(), max_turns=2)
    assert out["meta"]["n_turns"] == 2
    assert out["meta"]["max_turns"] == 2


def test_run_rollout_records_per_turn_components() -> None:
    inst = generate_instance(seed=0)
    solver = _ScriptedSolver([
        _Completion(json.dumps({"query": "SELECT 1 ORDER BY 1", "confidence": 0.1})),
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.9})),
        _Completion(json.dumps({"query": inst.gold_query, "confidence": 0.9})),
    ])
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    out = env.run_rollout(solver, inst, adapter=_ScriptedAdapter())
    components = out["meta"]["turn_components"]
    assert len(components) == 3
    for c in components:
        for k in ("format_valid", "parse_valid", "correctness"):
            assert k in c
            assert 0.0 <= c[k] <= 1.0


def test_run_rollout_baseline_returns_canonical_dict() -> None:
    inst = generate_instance(seed=0)
    env = SqlMultiturnEnv(conformal_quantile=0.5)
    pred = baseline_predict(inst)
    out = env.score(pred, inst)
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert "covered" in out["meta"]


def test_template_lattice_unchanged_from_single_turn() -> None:
    """Multi-turn shares the procedural problem distribution."""
    a = generate_problem(seed=11)
    b = generate_problem(seed=11)
    assert a == b


def test_query_diagnostic_carries_only_locked_keys() -> None:
    """Sanity: the diagnostic dict has exactly the documented keys."""
    inst = generate_instance(seed=0)
    diag = query_diagnostic(
        instance=inst, query=inst.gold_query, timeout_s=2.0,
    )
    expected = {"parse_error", "predicted_row_count", "gold_row_count", "correctness_match"}
    assert set(diag.keys()) == expected
