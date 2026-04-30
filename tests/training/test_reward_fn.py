"""Unit tests for ``verifiable_labs_envs.training.reward_fn``.

Targets the sparse-fourier-recovery env (n=256, k=10 by default) with
deterministic, CPU-only completions — no LLM, no GPU. The tests verify
the three-stage gating (parse_valid / format_valid / score), failure-mode
robustness, determinism, statistics tracking, and TRL-shaped batch IO.
"""
from __future__ import annotations

import json
import math
import time

import pytest

from verifiable_labs_envs.training import RewardStats, make_reward_fn

# ── helpers ────────────────────────────────────────────────────────────


def _zero_completion(k: int = 10, n: int = 256) -> str:
    """Match what the CLI's `_strip_internals(zero_agent_prediction)` produces."""
    return json.dumps(
        {
            "support_idx": list(range(min(k, n))),
            "support_amp_x1000": [0] * k,
        }
    )


def _ones_completion(k: int = 10, n: int = 256) -> str:
    """Non-trivial valid prediction: amplitude=1.0 at the first k indices."""
    return json.dumps(
        {
            "support_idx": list(range(min(k, n))),
            "support_amp_x1000": [1000] * k,
        }
    )


def _wrong_key_json() -> str:
    return json.dumps({"foo": "bar", "baz": [1, 2, 3]})


def _wrong_length_json() -> str:
    """JSON with right keys but wrong list length (k=10 expected, got 3)."""
    return json.dumps(
        {
            "support_idx": [0, 1, 2],
            "support_amp_x1000": [100, 200, 300],
        }
    )


def _out_of_range_json(k: int = 10, n: int = 256) -> str:
    """Indices outside [0, n) — should fail format check."""
    return json.dumps(
        {
            "support_idx": [n + 5] * k,
            "support_amp_x1000": [100] * k,
        }
    )


@pytest.fixture(scope="module")
def reward_fn():
    return make_reward_fn("sparse-fourier-recovery")


# ── tests ──────────────────────────────────────────────────────────────


def test_make_reward_fn_metadata(reward_fn) -> None:
    assert reward_fn.env_id == "sparse-fourier-recovery"
    assert reward_fn.seed_kwarg == "instance_seed"
    assert isinstance(reward_fn.stats, RewardStats)


def test_basic_batch_shape_and_finiteness(reward_fn) -> None:
    reward_fn.stats.reset()
    n = 50
    completions = [_zero_completion() for _ in range(n)]
    seeds = list(range(n))
    rewards = reward_fn(prompts=[""] * n, completions=completions, instance_seed=seeds)
    assert isinstance(rewards, list)
    assert len(rewards) == n
    assert all(isinstance(r, float) for r in rewards)
    assert all(math.isfinite(r) for r in rewards)


def test_empty_completion_returns_zero_no_crash(reward_fn) -> None:
    reward_fn.stats.reset()
    rewards = reward_fn(prompts=[""], completions=[""], instance_seed=[0])
    assert rewards == [0.0]
    assert reward_fn.stats.n_calls == 1
    assert reward_fn.stats.n_parse_valid == 0
    assert reward_fn.stats.n_format_valid == 0
    rec = reward_fn.stats.per_call[-1]
    assert rec["failure_type"] == "parse_error"
    assert rec["components"]["parse_valid"] == 0.0
    assert rec["components"]["format_valid"] == 0.0


def test_garbage_completion_returns_zero(reward_fn) -> None:
    reward_fn.stats.reset()
    completions = [
        "totally not json at all",
        "\x00\x01\x02 garbage bytes",
        "Here is my answer maybe perhaps",
    ]
    rewards = reward_fn(
        prompts=[""] * 3, completions=completions, instance_seed=[0, 1, 2]
    )
    assert rewards == [0.0, 0.0, 0.0]
    assert reward_fn.stats.n_parse_valid == 0
    for rec in reward_fn.stats.per_call:
        assert rec["failure_type"] == "parse_error"


def test_valid_json_wrong_keys_parse_valid_only(reward_fn) -> None:
    reward_fn.stats.reset()
    rewards = reward_fn(
        prompts=[""], completions=[_wrong_key_json()], instance_seed=[0]
    )
    assert rewards == [0.0]
    rec = reward_fn.stats.per_call[-1]
    assert rec["failure_type"] == "format_error"
    assert rec["components"]["parse_valid"] == 1.0
    assert rec["components"]["format_valid"] == 0.0
    assert reward_fn.stats.n_parse_valid == 1
    assert reward_fn.stats.n_format_valid == 0


def test_valid_json_wrong_length_parse_valid_only(reward_fn) -> None:
    reward_fn.stats.reset()
    rewards = reward_fn(
        prompts=[""], completions=[_wrong_length_json()], instance_seed=[0]
    )
    assert rewards == [0.0]
    rec = reward_fn.stats.per_call[-1]
    assert rec["components"]["parse_valid"] == 1.0
    assert rec["components"]["format_valid"] == 0.0
    assert rec["failure_type"] == "format_error"


def test_out_of_range_indices_parse_valid_only(reward_fn) -> None:
    reward_fn.stats.reset()
    rewards = reward_fn(
        prompts=[""], completions=[_out_of_range_json()], instance_seed=[0]
    )
    assert rewards == [0.0]
    rec = reward_fn.stats.per_call[-1]
    assert rec["components"]["parse_valid"] == 1.0
    assert rec["components"]["format_valid"] == 0.0


def test_zero_prediction_reward_in_expected_band(reward_fn) -> None:
    """Mirrors the M1 smoke run baseline (~0.336 mean)."""
    reward_fn.stats.reset()
    n = 5
    rewards = reward_fn(
        prompts=[""] * n,
        completions=[_zero_completion()] * n,
        instance_seed=list(range(n)),
    )
    assert all(0.32 <= r <= 0.36 for r in rewards), f"unexpected rewards: {rewards}"
    assert reward_fn.stats.n_parse_valid == n
    assert reward_fn.stats.n_format_valid == n
    for rec in reward_fn.stats.per_call:
        assert rec["components"]["parse_valid"] == 1.0
        assert rec["components"]["format_valid"] == 1.0
        # All three env-native components populated.
        for k in ("nmse", "support", "conformal"):
            assert isinstance(rec["components"][k], float)
            assert math.isfinite(rec["components"][k])


def test_determinism_bitwise_equality(reward_fn) -> None:
    """Same (seed, completion) twice must produce bitwise-identical rewards."""
    reward_fn.stats.reset()
    completion = _ones_completion()
    r1 = reward_fn(prompts=[""], completions=[completion], instance_seed=[42])
    r2 = reward_fn(prompts=[""], completions=[completion], instance_seed=[42])
    assert r1 == r2, f"non-deterministic: r1={r1!r} r2={r2!r}"
    # And bitwise via repr.
    assert repr(r1[0]) == repr(r2[0])


def test_mixed_batch_counter_breakdown(reward_fn) -> None:
    reward_fn.stats.reset()
    completions = [
        "garbage",  # parse fail
        _wrong_key_json(),  # parse ok, format fail
        _zero_completion(),  # all ok
        _zero_completion(),  # all ok
        _wrong_length_json(),  # parse ok, format fail
    ]
    rewards = reward_fn(
        prompts=[""] * 5, completions=completions, instance_seed=[0, 1, 2, 3, 4]
    )
    assert len(rewards) == 5
    assert rewards[0] == 0.0
    assert rewards[1] == 0.0
    assert rewards[2] > 0.0
    assert rewards[3] > 0.0
    assert rewards[4] == 0.0
    s = reward_fn.stats
    assert s.n_calls == 5
    assert s.n_parse_valid == 4  # all but the first
    assert s.n_format_valid == 2  # only the two zero predictions


def test_stats_aggregate_keys_and_rates(reward_fn) -> None:
    reward_fn.stats.reset()
    n = 4
    rewards = reward_fn(
        prompts=[""] * n,
        completions=[_zero_completion()] * n,
        instance_seed=list(range(n)),
    )
    agg = reward_fn.stats.aggregate()
    expected_keys = {
        "n_calls",
        "mean_reward",
        "parse_valid_rate",
        "format_valid_rate",
        "score_exception_rate",
        "mean_nmse",
        "mean_support",
        "mean_conformal",
    }
    assert set(agg) == expected_keys
    assert agg["n_calls"] == float(n)
    assert agg["parse_valid_rate"] == 1.0
    assert agg["format_valid_rate"] == 1.0
    assert agg["mean_reward"] == pytest.approx(sum(rewards) / n)


def test_per_call_records_preserve_input_order(reward_fn) -> None:
    reward_fn.stats.reset()
    seeds = [11, 22, 33]
    reward_fn(
        prompts=[""] * 3,
        completions=[_zero_completion()] * 3,
        instance_seed=seeds,
    )
    assert [rec["seed"] for rec in reward_fn.stats.per_call] == seeds


def test_missing_seed_column_raises(reward_fn) -> None:
    reward_fn.stats.reset()
    with pytest.raises(ValueError, match="instance_seed"):
        reward_fn(prompts=[""], completions=[_zero_completion()])


def test_mismatched_seed_completion_lengths_raise(reward_fn) -> None:
    reward_fn.stats.reset()
    with pytest.raises(ValueError, match="!="):
        reward_fn(
            prompts=[""] * 2,
            completions=[_zero_completion(), _zero_completion()],
            instance_seed=[0, 1, 2],  # length mismatch
        )


def test_50_calls_under_30s_cpu(reward_fn) -> None:
    reward_fn.stats.reset()
    n = 50
    completions = [_zero_completion()] * n
    seeds = list(range(n))
    t0 = time.perf_counter()
    rewards = reward_fn(prompts=[""] * n, completions=completions, instance_seed=seeds)
    elapsed = time.perf_counter() - t0
    assert len(rewards) == n
    assert elapsed < 30.0, f"50 calls took {elapsed:.2f}s, expected < 30s"
