"""Unit tests for ``verifiable_labs_envs.reward_distillation.consensus``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    borderline_indices,
    consensus_reward,
    disagreement,
    disagreement_metrics,
)


def test_default_weights_sum_to_one() -> None:
    assert pytest.approx(1.0) == DEFAULT_ENV_WEIGHT + DEFAULT_FRONTIER_WEIGHT


def test_consensus_env_only_passthrough() -> None:
    # Only env reward present (the common case) — return env_reward unchanged.
    assert consensus_reward(0.42, None) == pytest.approx(0.42)
    assert consensus_reward(0.0, None) == pytest.approx(0.0)
    assert consensus_reward(1.0, None) == pytest.approx(1.0)


def test_consensus_frontier_only_passthrough() -> None:
    # External rows have no env reward; consensus = frontier judgment.
    assert consensus_reward(None, 0.7) == pytest.approx(0.7)


def test_consensus_blend_70_30() -> None:
    # 0.7 * 0.8 + 0.3 * 0.2 = 0.56 + 0.06 = 0.62
    blended = consensus_reward(0.8, 0.2)
    assert blended == pytest.approx(0.62, abs=1e-9)


def test_consensus_clips_to_unit_interval() -> None:
    # Weighted blend cannot exceed the input range, but defensive clipping
    # protects against malformed external rows that exceed [0, 1].
    assert consensus_reward(1.5, None) == pytest.approx(1.0)
    assert consensus_reward(-0.5, None) == pytest.approx(0.0)


def test_consensus_rejects_both_none() -> None:
    with pytest.raises(ValueError, match="at least one"):
        consensus_reward(None, None)


def test_consensus_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        consensus_reward(0.5, 0.5, env_weight=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        consensus_reward(0.5, 0.5, frontier_weight=-0.1)


def test_consensus_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        consensus_reward(0.5, 0.5, env_weight=0.0, frontier_weight=0.0)


def test_consensus_custom_weights_50_50() -> None:
    blended = consensus_reward(1.0, 0.0, env_weight=0.5, frontier_weight=0.5)
    assert blended == pytest.approx(0.5)


def test_disagreement_basic() -> None:
    assert disagreement(0.8, 0.2) == pytest.approx(0.6)
    assert disagreement(0.5, 0.5) == pytest.approx(0.0)


def test_disagreement_metrics_empty_collection() -> None:
    metrics = disagreement_metrics([])
    assert metrics["count"] == 0.0
    assert metrics["mean"] == 0.0
    assert metrics["max"] == 0.0


def test_disagreement_metrics_handles_none_rows() -> None:
    rows = [
        {"disagreement": 0.1},
        {"disagreement": None},
        {"disagreement": 0.5},
        {"unrelated": "value"},  # row with no disagreement key — skipped
    ]
    metrics = disagreement_metrics(rows)
    assert metrics["count"] == 2.0
    assert metrics["mean"] == pytest.approx(0.3)
    assert metrics["max"] == pytest.approx(0.5)
    assert metrics["min"] == pytest.approx(0.1)


def test_disagreement_metrics_quantiles() -> None:
    # Disagreement values 0..1 in 0.1 steps (n=11) — quartiles fall on
    # interpolation boundaries; just sanity-check their order.
    rows = [{"disagreement": i / 10.0} for i in range(11)]
    metrics = disagreement_metrics(rows)
    assert metrics["count"] == 11.0
    assert metrics["min"] <= metrics["p25"] <= metrics["p50"] <= metrics["p75"] <= metrics["max"]
    assert metrics["p50"] == pytest.approx(0.5)


def test_borderline_indices_basic() -> None:
    rewards = [0.05, 0.32, 0.5, 0.65, 0.95]
    idxs = borderline_indices(rewards)
    # Default window is (0.3, 0.7) — 0.32, 0.5, 0.65 are inside.
    assert idxs == [1, 2, 3]


def test_borderline_indices_excludes_endpoints() -> None:
    # 0.3 and 0.7 are *open* interval bounds — not included.
    rewards = [0.3, 0.7]
    assert borderline_indices(rewards) == []


def test_borderline_indices_custom_window() -> None:
    rewards = [0.05, 0.15, 0.5, 0.85, 0.95]
    idxs = borderline_indices(rewards, low=0.1, high=0.9)
    assert idxs == [1, 2, 3]


def test_borderline_indices_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        borderline_indices([0.5], low=0.7, high=0.3)
    with pytest.raises(ValueError, match="0 <= low < high <= 1"):
        borderline_indices([0.5], low=-0.1, high=0.5)
