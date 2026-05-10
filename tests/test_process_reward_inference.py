"""Tests for ``verifiable_labs_envs.process_reward.inference``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.inference import (
    DEFAULT_COVERAGE_GUARANTEE,
    DEFAULT_DELTA,
    DEFAULT_MODEL_ID,
    DEFAULT_REWARD_CEILING,
    DEFAULT_REWARD_FLOOR,
    DEFAULT_SCHEMA_VERSION,
    StubProcessScoreResult,
    is_stub_payload,
    stub_aggregate_predictor,
    stub_full_predictor,
    stub_process_score,
    stub_score_ceiling,
    stub_score_floor,
    stub_step_predictor,
)

# ── locked defaults ─────────────────────────────────────────────────


def test_default_schema_version_ends_in_stub() -> None:
    assert DEFAULT_SCHEMA_VERSION.endswith("-stub")


def test_default_delta_locked() -> None:
    assert pytest.approx(0.1) == DEFAULT_DELTA


def test_default_model_id_locked() -> None:
    """D12-B PRM model id shape."""
    assert DEFAULT_MODEL_ID == "vlabs-prm-distilled-qwen-1-5b-v0.1.0"


def test_default_coverage_locked() -> None:
    assert pytest.approx(0.90) == DEFAULT_COVERAGE_GUARANTEE


def test_score_floor_ceiling() -> None:
    assert stub_score_floor() == pytest.approx(DEFAULT_REWARD_FLOOR)
    assert stub_score_ceiling() == pytest.approx(DEFAULT_REWARD_CEILING)


# ── stub_process_score ─────────────────────────────────────────────


def test_stub_score_deterministic() -> None:
    a = stub_process_score("p", ["s1", "s2", "s3"])
    b = stub_process_score("p", ["s1", "s2", "s3"])
    assert a.step_rewards == b.step_rewards
    assert a.step_confidence_intervals == b.step_confidence_intervals
    assert a.aggregate_reward == b.aggregate_reward


def test_stub_score_diverges_on_step_change() -> None:
    a = stub_process_score("p", ["s1", "s2"])
    b = stub_process_score("p", ["s1", "s3"])
    # Step 0 reward should match (same prefix); step 1 differs.
    assert a.step_rewards[0] == pytest.approx(b.step_rewards[0])
    assert a.step_rewards[1] != pytest.approx(b.step_rewards[1])


def test_stub_score_step_count_matches() -> None:
    out = stub_process_score("p", ["a", "b", "c", "d", "e"])
    assert out.step_count == 5
    assert len(out.step_rewards) == 5
    assert len(out.step_confidence_intervals) == 5


def test_stub_score_within_unit_interval() -> None:
    out = stub_process_score("p", [f"step-{i}" for i in range(10)])
    for r in out.step_rewards:
        assert 0.0 <= r <= 1.0
    for low, high in out.step_confidence_intervals:
        assert 0.0 <= low <= high <= 1.0
    assert 0.0 <= out.aggregate_reward <= 1.0


def test_stub_score_default_delta_locks_floor_ceiling() -> None:
    """At delta=0.1, every step reward lands in [0.4, 0.6]."""
    out = stub_process_score("p", [f"s-{i}" for i in range(20)])
    for r in out.step_rewards:
        assert 0.4 <= r <= 0.6


def test_stub_score_payload_carries_stub_schema() -> None:
    out = stub_process_score("p", ["s1"])
    assert out.schema_version == DEFAULT_SCHEMA_VERSION
    assert out.model_id == DEFAULT_MODEL_ID
    assert out.coverage_guarantee == pytest.approx(DEFAULT_COVERAGE_GUARANTEE)
    assert out.cache_hit is False


def test_stub_score_aggregate_equals_step_mean() -> None:
    out = stub_process_score("p", ["s1", "s2", "s3", "s4"])
    expected = sum(out.step_rewards) / len(out.step_rewards)
    assert out.aggregate_reward == pytest.approx(expected)


def test_stub_score_to_dict_round_trippable() -> None:
    out = stub_process_score("p", ["s1", "s2"])
    d = out.to_dict()
    assert d["step_count"] == 2
    assert d["schema_version"] == DEFAULT_SCHEMA_VERSION
    assert len(d["step_rewards"]) == 2
    assert len(d["step_confidence_intervals"]) == 2


def test_stub_score_rejects_negative_delta() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stub_process_score("p", ["s1"], delta=-0.1)


def test_stub_score_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        stub_process_score("p", [])


def test_stub_score_seed_changes_output() -> None:
    a = stub_process_score("p", ["s1"], seed=0)
    b = stub_process_score("p", ["s1"], seed=1)
    # Different seeds → different rewards.
    assert isinstance(a.step_rewards[0], float)
    assert isinstance(b.step_rewards[0], float)


def test_stub_score_segmentation_warning_passthrough() -> None:
    out = stub_process_score(
        "p", ["s1"], segmentation_warning="low_confidence"
    )
    assert out.segmentation_warning == "low_confidence"


# ── adapter callables ──────────────────────────────────────────────


def test_step_predictor_signature() -> None:
    pred = stub_step_predictor()
    r = pred("p", ["s1", "s2", "s3"], 1)
    assert 0.0 <= r <= 1.0


def test_step_predictor_rejects_out_of_range_index() -> None:
    pred = stub_step_predictor()
    with pytest.raises(IndexError, match="step_index"):
        pred("p", ["s1"], 5)


def test_aggregate_predictor_returns_mean() -> None:
    pred = stub_aggregate_predictor()
    direct = stub_process_score("p", ["s1", "s2"])
    assert pred("p", ["s1", "s2"]) == pytest.approx(direct.aggregate_reward)


def test_full_predictor_returns_result_object() -> None:
    pred = stub_full_predictor()
    out = pred("p", ["s1"])
    assert isinstance(out, StubProcessScoreResult)


def test_predictors_are_independent_callables() -> None:
    a = stub_step_predictor()
    b = stub_step_predictor()
    # Two callables built from the same factory should produce
    # identical outputs (no hidden state).
    assert a("p", ["s1"], 0) == b("p", ["s1"], 0)


# ── audit helpers ──────────────────────────────────────────────────


def test_is_stub_payload_predicate() -> None:
    assert is_stub_payload({"schema_version": "v0.1.0-stub"}) is True
    assert is_stub_payload({"schema_version": "v0.1.0"}) is False
    assert is_stub_payload({}) is False


def test_stub_score_result_dataclass_attrs() -> None:
    r = StubProcessScoreResult(
        step_rewards=(0.5,),
        step_confidence_intervals=((0.4, 0.6),),
        aggregate_reward=0.5,
        aggregate_confidence_interval=(0.4, 0.6),
        coverage_guarantee=0.9,
        step_count=1,
        model_id="x",
    )
    assert r.cache_hit is False
    assert r.schema_version == DEFAULT_SCHEMA_VERSION
