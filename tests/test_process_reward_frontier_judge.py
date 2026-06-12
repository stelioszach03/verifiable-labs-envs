"""Tests for ``verifiable_labs_envs.process_reward.frontier_judge``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.dataset import (
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
)
from verifiable_labs_envs.process_reward.frontier_judge import (
    DEFAULT_BORDERLINE_HIGH,
    DEFAULT_BORDERLINE_LOW,
    DEFAULT_COST_CAP_USD,
    PerStepFrontierResult,
    estimate_step_judge_cost,
    is_borderline_step,
    merge_per_step_judgments,
    openrouter_step_judge_caller,
    resolve_api_key,
    sample_per_step_judgments,
    select_borderline_step_targets,
    stub_step_judge_caller,
)

# ── helpers ─────────────────────────────────────────────────────────


def _trace(
    *,
    row_id: str,
    env_rewards: tuple[float | None, ...],
    n_steps: int | None = None,
) -> ProcessRewardTraceRow:
    n = n_steps if n_steps is not None else len(env_rewards)
    return ProcessRewardTraceRow(
        row_id=row_id,
        env_id="math-algebra",
        prompt=f"prompt for {row_id}",
        steps=tuple(f"step-{i}" for i in range(n)),
        step_rewards=env_rewards,
        step_components=tuple(None for _ in range(n)),
        step_conformal_intervals=tuple(None for _ in range(n)),
        step_frontier_judgments=tuple(None for _ in range(n)),
        step_frontier_rationales=tuple(None for _ in range(n)),
        step_consensus_rewards=tuple(
            (r if r is not None else 0.5) for r in env_rewards
        ),
        step_disagreements=tuple(None for _ in range(n)),
        aggregate_reward=0.5,
        aggregate_conformal_interval=None,
        decomposition="text_progress",
        segmentation_strategy="explicit_step_marker",
        segmentation_confidence=0.95,
        truncated=False,
        source="env",
        metadata={"schema_version": SCHEMA_VERSION},
    )


# ── locked constants ────────────────────────────────────────────────


def test_default_cost_cap_locked_per_plan() -> None:
    """Plan §5 D8 / §19: $50 cap per slice."""
    assert pytest.approx(50.0) == DEFAULT_COST_CAP_USD


def test_default_borderline_window_locked() -> None:
    assert pytest.approx(0.3) == DEFAULT_BORDERLINE_LOW
    assert pytest.approx(0.7) == DEFAULT_BORDERLINE_HIGH


# ── is_borderline_step ──────────────────────────────────────────────


def test_is_borderline_step_default_window() -> None:
    assert is_borderline_step(0.5) is True
    assert is_borderline_step(0.31) is True
    assert is_borderline_step(0.69) is True


def test_is_borderline_step_outside_window() -> None:
    assert is_borderline_step(0.05) is False
    assert is_borderline_step(0.95) is False


def test_is_borderline_step_excludes_endpoints() -> None:
    assert is_borderline_step(DEFAULT_BORDERLINE_LOW) is False
    assert is_borderline_step(DEFAULT_BORDERLINE_HIGH) is False


def test_is_borderline_step_handles_none() -> None:
    assert is_borderline_step(None) is False


def test_is_borderline_step_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        is_borderline_step(0.5, low=0.7, high=0.3)


# ── select_borderline_step_targets ──────────────────────────────────


def test_select_borderline_step_targets_basic() -> None:
    rows = [
        _trace(row_id="r0", env_rewards=(0.1, 0.5, 0.9)),
        _trace(row_id="r1", env_rewards=(0.4, 0.6)),
    ]
    selected = select_borderline_step_targets(rows, fraction=1.0)
    # 3 borderline steps total: (r0, 1), (r1, 0), (r1, 1).
    assert len(selected) == 3


def test_select_borderline_step_targets_zero_fraction_returns_at_least_one() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5, 0.5))]
    selected = select_borderline_step_targets(rows, fraction=0.0001)
    assert len(selected) == 1


def test_select_borderline_step_targets_no_borderline_returns_empty() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.0, 1.0, None))]
    assert select_borderline_step_targets(rows, fraction=1.0) == []


def test_select_borderline_step_targets_max_steps_caps() -> None:
    rows = [_trace(row_id="r", env_rewards=tuple(0.5 for _ in range(20)))]
    selected = select_borderline_step_targets(
        rows, fraction=1.0, max_steps=5
    )
    assert len(selected) == 5


def test_select_borderline_step_targets_seed_deterministic() -> None:
    rows = [
        _trace(
            row_id=f"r{i}",
            env_rewards=tuple(0.4 + 0.005 * j for j in range(10)),
        )
        for i in range(5)
    ]
    a = select_borderline_step_targets(rows, fraction=0.2, seed=42)
    b = select_borderline_step_targets(rows, fraction=0.2, seed=42)
    assert [(t[0].row_id, t[1]) for t in a] == [
        (t[0].row_id, t[1]) for t in b
    ]


def test_select_borderline_step_targets_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        select_borderline_step_targets([], fraction=1.5)


# ── stub caller ────────────────────────────────────────────────────


def test_stub_caller_returns_chat_completion_shape() -> None:
    payload = stub_step_judge_caller("p", "prefix", "step", "model", "key")
    assert "choices" in payload
    assert payload["choices"][0]["message"]["content"]


def test_stub_caller_score_is_05() -> None:
    import json

    payload = stub_step_judge_caller("p", "prefix", "step", "model", "key")
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    assert parsed["score"] == 0.5


# ── sample_per_step_judgments ──────────────────────────────────────


def test_sample_per_step_judgments_with_stub() -> None:
    rows = [
        _trace(row_id="r0", env_rewards=(0.5, 0.5, 0.5)),
        _trace(row_id="r1", env_rewards=(0.5,)),
    ]
    results = sample_per_step_judgments(
        rows,
        fraction=1.0,
        judge_caller=stub_step_judge_caller,
        api_key="<stub>",
    )
    assert len(results) == 4
    for r in results:
        assert isinstance(r, PerStepFrontierResult)
        assert r.parsed_ok is True
        assert r.score == 0.5


def test_sample_per_step_judgments_skips_non_borderline() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.0, 0.5, 1.0))]
    results = sample_per_step_judgments(
        rows,
        fraction=1.0,
        judge_caller=stub_step_judge_caller,
        api_key="<stub>",
    )
    # Only the middle step is borderline.
    assert len(results) == 1
    assert results[0].step_index == 1


def test_sample_per_step_judgments_requires_caller_or_key() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]
    with pytest.raises(ValueError, match="judge_caller or api_key"):
        sample_per_step_judgments(rows, fraction=1.0)


def test_sample_per_step_judgments_handles_caller_error() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    results = sample_per_step_judgments(
        rows, fraction=1.0, judge_caller=boom, api_key="<stub>"
    )
    assert len(results) == 1
    assert results[0].parsed_ok is False
    assert "api down" in results[0].rationale


def test_sample_per_step_judgments_raise_on_error() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError, match="api down"):
        sample_per_step_judgments(
            rows,
            fraction=1.0,
            judge_caller=boom,
            api_key="<stub>",
            raise_on_error=True,
        )


# ── merge_per_step_judgments ───────────────────────────────────────


def test_merge_per_step_judgments_applies_score() -> None:
    rows = [_trace(row_id="r0", env_rewards=(0.4, 0.5, 0.6))]
    judgments = [
        PerStepFrontierResult(
            row_id="r0",
            step_index=1,
            score=0.9,
            rationale="high quality",
            judge_model="m",
            raw_response='{"score":0.9,"rationale":"high quality"}',
            parsed_ok=True,
        )
    ]
    merged = merge_per_step_judgments(rows, judgments)
    assert merged[0].step_frontier_judgments[1] == pytest.approx(0.9)
    assert merged[0].source == "judgment"
    # Step 1 consensus = 0.7*0.5 + 0.3*0.9 = 0.62.
    assert merged[0].step_consensus_rewards[1] == pytest.approx(0.62)
    # Other steps fall back to env-only.
    assert merged[0].step_consensus_rewards[0] == pytest.approx(0.4)


def test_merge_skips_unparsed_results() -> None:
    rows = [_trace(row_id="r0", env_rewards=(0.5,))]
    judgments = [
        PerStepFrontierResult(
            row_id="r0",
            step_index=0,
            score=0.5,
            rationale="<failed>",
            judge_model="m",
            raw_response="",
            parsed_ok=False,
        )
    ]
    merged = merge_per_step_judgments(rows, judgments)
    assert merged[0] == rows[0]


def test_merge_preserves_unjudged_rows() -> None:
    rows = [
        _trace(row_id="r0", env_rewards=(0.5,)),
        _trace(row_id="r1", env_rewards=(0.5,)),
    ]
    judgments = [
        PerStepFrontierResult(
            row_id="r0",
            step_index=0,
            score=0.7,
            rationale="ok",
            judge_model="m",
            raw_response="{}",
            parsed_ok=True,
        )
    ]
    merged = merge_per_step_judgments(rows, judgments)
    assert merged[1] == rows[1]


def test_merge_updates_aggregate_and_disagreement() -> None:
    rows = [_trace(row_id="r0", env_rewards=(0.4, 0.6))]
    judgments = [
        PerStepFrontierResult(
            row_id="r0",
            step_index=0,
            score=1.0,
            rationale="x",
            judge_model="m",
            raw_response="{}",
            parsed_ok=True,
        )
    ]
    merged = merge_per_step_judgments(rows, judgments)
    assert merged[0].step_disagreements[0] == pytest.approx(abs(0.4 - 1.0))


# ── parsing edge cases ──────────────────────────────────────────────


def test_parsing_strict_json() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def caller(*args, **kwargs):
        return {
            "choices": [
                {"message": {"content": '{"score": 0.85, "rationale": "good"}'}}
            ]
        }

    results = sample_per_step_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].score == pytest.approx(0.85)
    assert results[0].parsed_ok is True


def test_parsing_handles_chatty_response() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def caller(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            'Sure!\n{"score": 0.42, "rationale": "x"}\nOk.'
                        )
                    }
                }
            ]
        }

    results = sample_per_step_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].score == pytest.approx(0.42)


def test_parsing_clips_to_unit_interval() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def caller(*args, **kwargs):
        return {
            "choices": [
                {"message": {"content": '{"score": 1.5, "rationale": "x"}'}}
            ]
        }

    results = sample_per_step_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].score == 1.0


def test_parsing_falls_back_on_unparseable() -> None:
    rows = [_trace(row_id="r", env_rewards=(0.5,))]

    def caller(*args, **kwargs):
        return {"choices": [{"message": {"content": "not json"}}]}

    results = sample_per_step_judgments(
        rows, fraction=1.0, judge_caller=caller, api_key="<stub>"
    )
    assert results[0].parsed_ok is False
    assert results[0].score == 0.5


# ── openrouter caller / cost / api key ─────────────────────────────


def test_openrouter_caller_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        openrouter_step_judge_caller("p", "", "s", "m", "")


def test_estimate_step_judge_cost_default() -> None:
    assert estimate_step_judge_cost(100) == pytest.approx(0.5)
    assert estimate_step_judge_cost(0) == pytest.approx(0.0)


def test_estimate_step_judge_cost_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_step_judge_cost(-1)


def test_resolve_api_key_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert resolve_api_key() is None
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-test ")
    assert resolve_api_key() == "sk-test"
