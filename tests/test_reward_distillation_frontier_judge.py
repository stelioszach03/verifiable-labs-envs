"""Unit tests for ``verifiable_labs_envs.reward_distillation.frontier_judge``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.reward_distillation.dataset import (
    SCHEMA_VERSION,
    RewardTrainingRow,
)
from verifiable_labs_envs.reward_distillation.frontier_judge import (
    DEFAULT_BORDERLINE_HIGH,
    DEFAULT_BORDERLINE_LOW,
    FrontierJudgeResult,
    estimate_judge_cost,
    is_borderline,
    merge_judgments,
    openrouter_judge_caller,
    resolve_api_key,
    sample_frontier_judgments,
    select_borderline_rows,
    stub_judge_caller,
)

# ── helpers ─────────────────────────────────────────────────────────


def _row(env_reward: float | None, idx: int) -> RewardTrainingRow:
    return RewardTrainingRow(
        row_id=f"rwd_{idx:016x}",
        env_id="math-algebra",
        prompt=f"prompt-{idx}",
        completion=f"completion-{idx}",
        env_reward=env_reward,
        env_components=None,
        conformal_interval=None,
        frontier_judgment=None,
        frontier_rationale=None,
        consensus_reward=env_reward if env_reward is not None else 0.5,
        disagreement=None,
        source="env",
        metadata={"schema_version": SCHEMA_VERSION},
    )


# ── is_borderline ───────────────────────────────────────────────────


def test_is_borderline_default_window() -> None:
    assert is_borderline(0.5) is True
    assert is_borderline(0.31) is True
    assert is_borderline(0.69) is True


def test_is_borderline_outside_window() -> None:
    assert is_borderline(0.05) is False
    assert is_borderline(0.95) is False


def test_is_borderline_excludes_endpoints() -> None:
    """Open interval — exact bounds are NOT borderline."""
    assert is_borderline(DEFAULT_BORDERLINE_LOW) is False
    assert is_borderline(DEFAULT_BORDERLINE_HIGH) is False


def test_is_borderline_handles_none() -> None:
    assert is_borderline(None) is False


def test_is_borderline_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        is_borderline(0.5, low=0.7, high=0.3)


# ── select_borderline_rows ──────────────────────────────────────────


def test_select_borderline_rows_default_fraction() -> None:
    rows = [_row(reward, i) for i, reward in enumerate([0.1, 0.4, 0.5, 0.6, 0.9])]
    selected = select_borderline_rows(rows, fraction=1.0)
    assert len(selected) == 3
    assert all(0.3 < r.env_reward < 0.7 for r in selected)


def test_select_borderline_rows_zero_fraction_returns_at_least_one() -> None:
    """Spec: if any borderline rows exist, sample at least 1."""
    rows = [_row(0.5, 0), _row(0.4, 1)]
    selected = select_borderline_rows(rows, fraction=0.001)
    assert len(selected) == 1


def test_select_borderline_rows_no_borderline_returns_empty() -> None:
    rows = [_row(0.0, 0), _row(1.0, 1), _row(None, 2)]
    assert select_borderline_rows(rows, fraction=1.0) == []


def test_select_borderline_rows_max_rows_caps() -> None:
    rows = [_row(0.5, i) for i in range(20)]
    selected = select_borderline_rows(rows, fraction=1.0, max_rows=5)
    assert len(selected) == 5


def test_select_borderline_rows_seed_deterministic() -> None:
    rows = [_row(0.4 + 0.005 * i, i) for i in range(50)]
    a = select_borderline_rows(rows, fraction=0.2, seed=99)
    b = select_borderline_rows(rows, fraction=0.2, seed=99)
    assert [r.row_id for r in a] == [r.row_id for r in b]


def test_select_borderline_rows_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        select_borderline_rows([], fraction=2.0)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        select_borderline_rows([], fraction=-0.1)


# ── stub_judge_caller ───────────────────────────────────────────────


def test_stub_judge_caller_returns_chat_completion_shape() -> None:
    payload = stub_judge_caller("p", "c", "test-model", "key")
    assert "choices" in payload
    assert isinstance(payload["choices"], list)
    assert len(payload["choices"]) == 1
    msg = payload["choices"][0]["message"]
    assert "content" in msg


def test_stub_judge_caller_score_is_05() -> None:
    """Stub returns a uniform 0.5 — a sentinel value detectable in audits."""
    import json

    payload = stub_judge_caller("p", "c", "test-model", "key")
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    assert parsed["score"] == 0.5


# ── sample_frontier_judgments ───────────────────────────────────────


def test_sample_frontier_judgments_with_stub() -> None:
    rows = [_row(0.5, i) for i in range(10)]
    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=stub_judge_caller, api_key="<stub>"
    )
    assert len(results) == 10
    for result in results:
        assert isinstance(result, FrontierJudgeResult)
        assert result.parsed_ok is True
        assert result.score == 0.5


def test_sample_frontier_judgments_skips_non_borderline() -> None:
    rows = [_row(0.0, 0), _row(0.5, 1), _row(1.0, 2)]
    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=stub_judge_caller, api_key="<stub>"
    )
    # Only the middle row is borderline.
    assert len(results) == 1
    assert results[0].row_id == rows[1].row_id


def test_sample_frontier_judgments_requires_caller_or_key() -> None:
    rows = [_row(0.5, 0)]
    with pytest.raises(ValueError, match="judge_caller or api_key"):
        sample_frontier_judgments(rows, fraction=1.0)


def test_sample_frontier_judgments_handles_caller_error() -> None:
    rows = [_row(0.5, 0)]

    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=boom, api_key="<stub>"
    )
    assert len(results) == 1
    assert results[0].parsed_ok is False
    assert "api down" in results[0].rationale


def test_sample_frontier_judgments_raise_on_error() -> None:
    rows = [_row(0.5, 0)]

    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError, match="api down"):
        sample_frontier_judgments(
            rows,
            fraction=1.0,
            judge_caller=boom,
            api_key="<stub>",
            raise_on_error=True,
        )


# ── merge_judgments ─────────────────────────────────────────────────


def test_merge_judgments_applies_score_and_marks_source() -> None:
    rows = [_row(0.4, 0), _row(0.05, 1)]
    judgment = FrontierJudgeResult(
        row_id=rows[0].row_id,
        score=0.9,
        rationale="high quality",
        judge_model="test-model",
        raw_response='{"score":0.9,"rationale":"high quality"}',
        parsed_ok=True,
    )
    merged = merge_judgments(rows, [judgment])
    assert merged[0].frontier_judgment == 0.9
    assert merged[0].source == "judgment"
    assert merged[0].disagreement is not None
    assert merged[0].disagreement == pytest.approx(abs(0.4 - 0.9))
    # The 70/30 blend: 0.7 * 0.4 + 0.3 * 0.9 = 0.55
    assert merged[0].consensus_reward == pytest.approx(0.55)
    assert merged[0].metadata["judge_model"] == "test-model"
    # Untouched rows pass through unchanged.
    assert merged[1] == rows[1]


def test_merge_judgments_skips_unparsed_results() -> None:
    rows = [_row(0.5, 0)]
    judgment = FrontierJudgeResult(
        row_id=rows[0].row_id,
        score=0.5,
        rationale="<failed>",
        judge_model="test-model",
        raw_response="",
        parsed_ok=False,
    )
    merged = merge_judgments(rows, [judgment])
    assert merged[0] == rows[0]


def test_merge_judgments_preserves_order() -> None:
    rows = [_row(0.5, i) for i in range(5)]
    judgments = [
        FrontierJudgeResult(
            row_id=rows[2].row_id,
            score=0.7,
            rationale="ok",
            judge_model="m",
            raw_response="{}",
            parsed_ok=True,
        ),
    ]
    merged = merge_judgments(rows, judgments)
    assert [r.row_id for r in merged] == [r.row_id for r in rows]
    assert merged[2].frontier_judgment == 0.7
    for i in (0, 1, 3, 4):
        assert merged[i].frontier_judgment is None


# ── _extract_score_rationale parsing ────────────────────────────────


def test_parsing_strict_json() -> None:
    rows = [_row(0.5, 0)]

    def caller(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"score": 0.85, "rationale": "good"}'
                    }
                }
            ]
        }

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].parsed_ok is True
    assert results[0].score == 0.85
    assert results[0].rationale == "good"


def test_parsing_extracts_json_block_from_chatty_response() -> None:
    rows = [_row(0.5, 0)]

    def caller(*args, **kwargs):
        chatty = (
            "Sure! Here is the score:\n"
            '{"score": 0.42, "rationale": "borderline"}\n'
            "Hope this helps."
        )
        return {"choices": [{"message": {"content": chatty}}]}

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].parsed_ok is True
    assert results[0].score == pytest.approx(0.42)


def test_parsing_clips_to_unit_interval() -> None:
    rows = [_row(0.5, 0)]

    def caller(*args, **kwargs):
        return {"choices": [{"message": {"content": '{"score": 1.5, "rationale": "x"}'}}]}

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].score == 1.0


def test_parsing_falls_back_on_unparseable() -> None:
    rows = [_row(0.5, 0)]

    def caller(*args, **kwargs):
        return {"choices": [{"message": {"content": "not json at all"}}]}

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].parsed_ok is False
    assert results[0].score == 0.5  # neutral fallback


def test_parsing_handles_empty_choices() -> None:
    rows = [_row(0.5, 0)]

    def caller(*args, **kwargs):
        return {"choices": []}

    results = sample_frontier_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].parsed_ok is False


# ── openrouter_judge_caller ─────────────────────────────────────────


def test_openrouter_caller_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        openrouter_judge_caller("p", "c", "model", "")


# ── cost + key helpers ──────────────────────────────────────────────


def test_estimate_judge_cost_default_per_row() -> None:
    assert estimate_judge_cost(100) == pytest.approx(0.5)
    assert estimate_judge_cost(0) == pytest.approx(0.0)


def test_estimate_judge_cost_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_judge_cost(-1)


def test_estimate_judge_cost_custom_rate() -> None:
    assert estimate_judge_cost(10, per_row_usd=0.01) == pytest.approx(0.1)


def test_resolve_api_key_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_api_key() is None


def test_resolve_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert resolve_api_key() is None
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-or-test ")
    assert resolve_api_key() == "sk-or-test"


def test_resolve_api_key_custom_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_ALT_KEY", "hello")
    assert resolve_api_key("MY_ALT_KEY") == "hello"
