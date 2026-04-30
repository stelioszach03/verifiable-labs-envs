"""Tests for ``adaptive_difficulty`` (RLVE Algorithm-1 tracker + anchor table)."""
from __future__ import annotations

import random

import pytest

from verifiable_labs_envs import load_environment
from verifiable_labs_envs.training.adaptive_difficulty import (
    ANCHOR_TABLES,
    AdaptiveDifficultyTracker,
    difficulty_to_kwargs,
    max_difficulty,
)

# ── tracker init ─────────────────────────────────────────────────────


def test_init_defaults() -> None:
    t = AdaptiveDifficultyTracker(env_id="sparse-fourier-recovery")
    assert t.l == 0
    assert t.h == 0
    assert t.a == 0
    assert t.b == 0
    assert t.advances == 0
    assert t.rollouts_total == 0
    assert t.tau_acc == 0.9
    assert t.tau_num == 32
    assert t.d_delta == 4
    assert t.success_threshold == 0.5


def test_init_invalid_tau_acc_raises() -> None:
    with pytest.raises(ValueError, match="tau_acc"):
        AdaptiveDifficultyTracker(env_id="x", tau_acc=0.0)
    with pytest.raises(ValueError, match="tau_acc"):
        AdaptiveDifficultyTracker(env_id="x", tau_acc=1.5)


def test_init_invalid_tau_num_raises() -> None:
    with pytest.raises(ValueError, match="tau_num"):
        AdaptiveDifficultyTracker(env_id="x", tau_num=0)


def test_init_invalid_d_delta_raises() -> None:
    with pytest.raises(ValueError, match="d_delta"):
        AdaptiveDifficultyTracker(env_id="x", d_delta=0)


# ── sample_difficulty ────────────────────────────────────────────────


def test_sample_difficulty_in_range_when_h_zero() -> None:
    t = AdaptiveDifficultyTracker(env_id="x")
    rng = random.Random(0)
    samples = [t.sample_difficulty(rng) for _ in range(50)]
    assert all(s == 0 for s in samples)


def test_sample_difficulty_uniform_in_window() -> None:
    t = AdaptiveDifficultyTracker(env_id="x")
    t.l = 2
    t.h = 6
    rng = random.Random(0)
    samples = [t.sample_difficulty(rng) for _ in range(2000)]
    assert all(2 <= s <= 6 for s in samples)
    counts = {d: samples.count(d) for d in range(2, 7)}
    # Each of 5 buckets should be roughly 400 with N=2000; allow generous slack.
    for d, c in counts.items():
        assert 250 <= c <= 550, f"bucket {d} count {c} outside [250, 550]"


# ── record_rollout + maybe_advance ───────────────────────────────────


def test_record_rollout_only_counts_at_h() -> None:
    t = AdaptiveDifficultyTracker(env_id="x")
    t.h = 3
    t.l = 0
    # Below ceiling: doesn't bump a/b
    for _ in range(10):
        t.record_rollout(difficulty=0, reward=1.0)
        t.record_rollout(difficulty=2, reward=1.0)
    assert t.a == 0
    assert t.b == 0
    assert t.rollouts_total == 20
    # At ceiling: bumps b and a
    for _ in range(7):
        t.record_rollout(difficulty=3, reward=1.0)
    assert t.b == 7
    assert t.a == 7  # all successes
    for _ in range(3):
        t.record_rollout(difficulty=3, reward=0.1)  # below success_threshold
    assert t.b == 10
    assert t.a == 7  # only 7 successes


def test_advance_when_acc_above_threshold() -> None:
    t = AdaptiveDifficultyTracker(env_id="x", tau_num=10, tau_acc=0.9)
    # 10 rollouts at h=0, 9 successes, 1 failure → 0.9 ≥ 0.9 → advance
    for _ in range(9):
        t.record_rollout(difficulty=0, reward=1.0)
    t.record_rollout(difficulty=0, reward=0.0)
    assert t.maybe_advance() is True
    assert t.h == 1
    assert t.l == 0
    assert t.advances == 1
    # Counters reset.
    assert t.a == 0
    assert t.b == 0


def test_no_advance_when_below_threshold() -> None:
    t = AdaptiveDifficultyTracker(env_id="x", tau_num=10, tau_acc=0.9)
    # 8 successes / 10 → 0.8 < 0.9 → no advance, but counters still reset
    for _ in range(8):
        t.record_rollout(difficulty=0, reward=1.0)
    for _ in range(2):
        t.record_rollout(difficulty=0, reward=0.0)
    assert t.maybe_advance() is False
    assert t.h == 0
    assert t.advances == 0
    assert t.a == 0  # reset
    assert t.b == 0


def test_no_advance_when_b_below_tau_num() -> None:
    t = AdaptiveDifficultyTracker(env_id="x", tau_num=32, tau_acc=0.5)
    # Only 5 rollouts — far below tau_num
    for _ in range(5):
        t.record_rollout(difficulty=0, reward=1.0)
    assert t.maybe_advance() is False
    # Counters NOT reset because the check short-circuited.
    assert t.b == 5
    assert t.a == 5


def test_sliding_window_clamps_l_when_over_d_delta() -> None:
    t = AdaptiveDifficultyTracker(env_id="x", tau_num=2, tau_acc=0.5, d_delta=3)
    # Force advancement many times; once h - l + 1 > 3, l should bump.
    for _ in range(6):
        t.record_rollout(difficulty=t.h, reward=1.0)
        t.record_rollout(difficulty=t.h, reward=1.0)
        t.maybe_advance()
    # 6 advances → h=6, d_delta=3 → l = h - d_delta + 1 = 4
    assert t.h == 6
    assert t.l == 4
    assert t.h - t.l + 1 == 3 == t.d_delta


# ── difficulty_to_kwargs ─────────────────────────────────────────────


def test_difficulty_to_kwargs_snap_to_lowest_anchor() -> None:
    out = difficulty_to_kwargs("sparse-fourier-recovery", 0)
    assert out == {"n": 64, "m": 16, "k": 3, "sigma": 0.05, "alpha": 0.1}


def test_difficulty_to_kwargs_snap_to_anchor_below() -> None:
    # Anchors at 0/5/10/15/20. Difficulty 7 snaps DOWN to 5.
    out = difficulty_to_kwargs("sparse-fourier-recovery", 7)
    assert out == {"n": 96, "m": 24, "k": 5, "sigma": 0.05, "alpha": 0.1}


def test_difficulty_to_kwargs_clamp_excess_to_top_anchor() -> None:
    out = difficulty_to_kwargs("sparse-fourier-recovery", 999)
    assert out["n"] == 256
    assert out["k"] == 20


def test_difficulty_to_kwargs_clamp_negative_to_zero() -> None:
    out = difficulty_to_kwargs("sparse-fourier-recovery", -5)
    assert out["k"] == 3
    assert out["n"] == 64


def test_difficulty_to_kwargs_unknown_env_returns_empty() -> None:
    assert difficulty_to_kwargs("does-not-exist", 10) == {}


def test_difficulty_to_kwargs_does_not_mutate_anchor() -> None:
    out = difficulty_to_kwargs("sparse-fourier-recovery", 0)
    out["n"] = 999  # mutate the returned dict
    fresh = difficulty_to_kwargs("sparse-fourier-recovery", 0)
    assert fresh["n"] == 64


def test_max_difficulty() -> None:
    assert max_difficulty("sparse-fourier-recovery") == 20
    assert max_difficulty("does-not-exist") == 0


# ── env integration ──────────────────────────────────────────────────


def test_env_generate_instance_honours_difficulty_kwargs() -> None:
    env = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)
    cases = [(0, 64, 3), (5, 96, 5), (10, 128, 8), (15, 192, 12), (20, 256, 20)]
    for difficulty, expected_n, expected_k in cases:
        kwargs = difficulty_to_kwargs("sparse-fourier-recovery", difficulty)
        instance = env.generate_instance(seed=0, **kwargs)
        assert instance.n == expected_n
        assert instance.k == expected_k
        # Spot-check shape consistency:
        assert len(instance.x_true) == expected_n
        assert len(instance.support_true) == expected_k
        assert instance.support_true.shape == (expected_k,)


# ── serialisation ────────────────────────────────────────────────────


def test_serialization_roundtrip() -> None:
    t = AdaptiveDifficultyTracker(
        env_id="sparse-fourier-recovery", tau_num=8, tau_acc=0.75
    )
    for _ in range(10):
        t.record_rollout(difficulty=0, reward=1.0)
    t.maybe_advance()
    t.record_rollout(difficulty=t.h, reward=1.0)

    d = t.to_dict()
    t2 = AdaptiveDifficultyTracker.from_dict(d)
    assert t2.env_id == t.env_id
    assert t2.l == t.l
    assert t2.h == t.h
    assert t2.a == t.a
    assert t2.b == t.b
    assert t2.advances == t.advances
    assert t2.rollouts_total == t.rollouts_total
    assert t2.tau_acc == t.tau_acc
    assert t2.tau_num == t.tau_num


# ── stats sanity ─────────────────────────────────────────────────────


def test_stats_contains_expected_keys() -> None:
    t = AdaptiveDifficultyTracker(env_id="x")
    s = t.stats()
    for k in ("env_id", "l", "h", "a", "b", "advances", "rollouts_total",
              "current_success_rate", "tau_acc", "tau_num", "d_delta"):
        assert k in s
    assert s["current_success_rate"] is None  # b=0


def test_anchor_tables_keys() -> None:
    """Sanity: ANCHOR_TABLES is a dict and contains sparse-fourier-recovery."""
    assert isinstance(ANCHOR_TABLES, dict)
    assert "sparse-fourier-recovery" in ANCHOR_TABLES
    anchors = ANCHOR_TABLES["sparse-fourier-recovery"]
    # Anchors must be sorted by threshold.
    thresholds = [t for t, _ in anchors]
    assert thresholds == sorted(thresholds)
