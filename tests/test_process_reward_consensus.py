"""Tests for ``verifiable_labs_envs.process_reward.consensus``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    borderline_step_indices,
    per_step_consensus,
    per_step_disagreement,
    per_step_disagreement_metrics,
    trace_aggregate_consensus,
)

# ── locked weights ──────────────────────────────────────────────────


def test_weights_inherited_from_phase29() -> None:
    """Plan §5 D5-D: the per-step blend reuses the 70/30 split."""
    assert pytest.approx(0.7) == DEFAULT_ENV_WEIGHT
    assert pytest.approx(0.3) == DEFAULT_FRONTIER_WEIGHT


# ── per_step_consensus ──────────────────────────────────────────────


def test_per_step_consensus_env_only_passthrough() -> None:
    """No frontier judgments: per-step consensus = env per-step."""
    env = [0.2, 0.4, 0.7]
    out = per_step_consensus(env)
    assert out == (pytest.approx(0.2), pytest.approx(0.4), pytest.approx(0.7))


def test_per_step_consensus_70_30_blend() -> None:
    env = [1.0, 0.0]
    front = [0.0, 1.0]
    out = per_step_consensus(env, front)
    # 0.7 * 1.0 + 0.3 * 0.0 = 0.7; 0.7 * 0.0 + 0.3 * 1.0 = 0.3.
    assert out[0] == pytest.approx(0.7)
    assert out[1] == pytest.approx(0.3)


def test_per_step_consensus_handles_none_env() -> None:
    """env None + frontier present → frontier value."""
    env = [None, 0.5]
    front = [0.8, 0.5]
    out = per_step_consensus(env, front)
    assert out[0] == pytest.approx(0.8)


def test_per_step_consensus_double_none_yields_neutral() -> None:
    out = per_step_consensus([None], [None])
    assert out[0] == pytest.approx(0.5)


def test_per_step_consensus_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        per_step_consensus([0.5], [0.5, 0.5])


# ── trace_aggregate_consensus ───────────────────────────────────────


def test_trace_aggregate_mean() -> None:
    assert trace_aggregate_consensus([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_trace_aggregate_empty_returns_neutral() -> None:
    assert trace_aggregate_consensus([]) == pytest.approx(0.5)


def test_trace_aggregate_env_blend() -> None:
    """env_blend method: 70/30 between env_outcome and per-step mean."""
    aggregate = trace_aggregate_consensus(
        [0.0, 0.0],          # mean = 0
        env_outcome=1.0,     # env outcome = 1
        method="env_blend",
    )
    # 0.7 * 1.0 + 0.3 * 0.0 = 0.7.
    assert aggregate == pytest.approx(0.7)


def test_trace_aggregate_env_blend_without_env_falls_back_to_mean() -> None:
    aggregate = trace_aggregate_consensus(
        [0.4, 0.6], env_outcome=None, method="env_blend"
    )
    assert aggregate == pytest.approx(0.5)


def test_trace_aggregate_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        trace_aggregate_consensus([0.5], method="bogus")


# ── per_step_disagreement ───────────────────────────────────────────


def test_per_step_disagreement_basic() -> None:
    out = per_step_disagreement([0.8, 0.2], [0.2, 0.5])
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.3)


def test_per_step_disagreement_handles_none_entries() -> None:
    out = per_step_disagreement([None, 0.5], [0.5, None])
    assert out == (None, None)


def test_per_step_disagreement_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        per_step_disagreement([0.5], [0.5, 0.5])


# ── per_step_disagreement_metrics ───────────────────────────────────


def test_metrics_empty() -> None:
    metrics = per_step_disagreement_metrics([])
    assert metrics["count"] == 0.0
    assert metrics["trace_count"] == 0.0


def test_metrics_aggregate_across_traces() -> None:
    rows = [
        {"step_disagreements": [0.1, 0.2, None]},
        {"step_disagreements": [0.5]},
        {"step_disagreements": None},
        {"unrelated_field": "skip"},
    ]
    metrics = per_step_disagreement_metrics(rows)
    assert metrics["count"] == 3.0
    assert metrics["trace_count"] == 2.0
    assert metrics["mean"] == pytest.approx((0.1 + 0.2 + 0.5) / 3)
    assert metrics["max"] == pytest.approx(0.5)
    assert metrics["min"] == pytest.approx(0.1)


def test_metrics_quantiles_ordered() -> None:
    rows = [
        {"step_disagreements": [i / 10.0 for i in range(11)]}
    ]
    metrics = per_step_disagreement_metrics(rows)
    assert metrics["min"] <= metrics["p50"] <= metrics["p90"] <= metrics["max"]


# ── borderline_step_indices ─────────────────────────────────────────


def test_borderline_step_indices_default_window() -> None:
    rewards = [0.05, 0.32, 0.5, 0.65, 0.95, None]
    idxs = borderline_step_indices(rewards)
    assert idxs == [1, 2, 3]


def test_borderline_step_indices_custom_window() -> None:
    idxs = borderline_step_indices(
        [0.15, 0.5, 0.85], low=0.1, high=0.9
    )
    assert idxs == [0, 1, 2]


def test_borderline_step_indices_excludes_endpoints() -> None:
    """0.3 / 0.7 are the open-interval bounds — not borderline."""
    assert borderline_step_indices([0.3, 0.7]) == []


def test_borderline_step_indices_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        borderline_step_indices([0.5], low=0.7, high=0.3)
