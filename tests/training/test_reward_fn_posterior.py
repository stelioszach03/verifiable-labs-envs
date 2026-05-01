"""Tests for ``make_reward_fn_posterior`` (P-GRPO posterior gating).

The first eight tests exercise the pure formula via ``posterior_reward``
with explicit numerical inputs, including the **critical posterior-gating
distinction test**. The remaining three test the TRL-shaped wrapper
end-to-end against the real ``sparse-fourier-recovery`` env.
"""
from __future__ import annotations

import json
import math
import time

import pytest

from verifiable_labs_envs import load_environment
from verifiable_labs_envs.training import (
    PosteriorRewardStats,
    make_reward_fn_posterior,
    posterior_reward,
)

ENV_ID = "sparse-fourier-recovery"


# ── helpers (identical to the M2 test file so the shapes match) ─────


def _zero_completion(k: int = 10, n: int = 256) -> str:
    return json.dumps(
        {
            "support_idx": list(range(min(k, n))),
            "support_amp_x1000": [0] * k,
        }
    )


def _oracle_completion(seed: int) -> str:
    env = load_environment(ENV_ID, calibration_quantile=2.0)
    inst = env.generate_instance(seed=seed)
    support = sorted(int(i) for i in inst.support_true)
    amps = [int(round(float(inst.x_true[i]) * 1000)) for i in support]
    return json.dumps({"support_idx": support, "support_amp_x1000": amps})


# ── pure formula tests (1–8) ─────────────────────────────────────────


def test_posterior_reward_format_fail_zeros_reward() -> None:
    """parse_valid=0 OR format_valid=0 → R=0 even with otherwise-perfect metrics.

    This is the R_format gate at work: r_format multiplies the entire
    expression, so a malformed completion can never earn quality or outcome
    credit by accident.
    """
    r, c = posterior_reward(
        parse_valid=0.0, format_valid=0.0,
        nmse_raw=0.0, support=1.0, conformal=1.0,  # all other signals "perfect"
    )
    assert r == 0.0
    assert c["r_format"] == 0.0
    # r_outcome is still computed as a TELEMETRY field (so we can audit),
    # but it doesn't affect the final reward thanks to the r_format gate.
    assert c["r_outcome"] == 1.0
    assert c["r_quality"] == 1.0
    assert c["r_unnormalised"] == 0.0  # 0 × (1 + 1 + 1×1) = 0
    assert c["r_normalised"] == 0.0


def test_posterior_reward_parse_only_no_format_zeros_reward() -> None:
    """Even if parse_valid=1 but format_valid=0, the gate still fires."""
    r, _ = posterior_reward(
        parse_valid=1.0, format_valid=0.0,
        nmse_raw=0.0, support=1.0, conformal=1.0,
    )
    assert r == 0.0


def test_posterior_reward_outcome_fail_only_format() -> None:
    """format_ok + outcome_fail (random support, high nmse) → ~1/3."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.5, support=0.0, conformal=0.5,
    )
    assert c["r_format"] == 1.0
    assert c["r_outcome"] == 0.0
    expected = 1.0 / 3.0
    assert abs(r - expected) < 1e-12, f"got {r}, expected {expected}"


def test_posterior_reward_outcome_pass_low_quality() -> None:
    """format_ok + outcome_pass + low_quality (boundary case)."""
    # nmse_raw=0.10, support=0.5, conformal=0.0 (low quality but outcome passes)
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.10, support=0.5, conformal=0.0,
    )
    assert c["r_format"] == 1.0
    assert c["r_outcome"] == 1.0
    expected_q = (1.0 - 0.10) * 0.5 + 0.0 * 0.5  # 0.45
    expected_r = (1.0 + 1.0 + 0.45) / 3.0  # ≈ 0.8167
    assert abs(c["r_quality"] - expected_q) < 1e-12
    assert abs(r - expected_r) < 1e-12


def test_posterior_reward_outcome_pass_high_quality() -> None:
    """format_ok + outcome_pass + high_quality → ~0.97."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.02, support=0.95, conformal=0.95,
    )
    assert c["r_format"] == 1.0
    assert c["r_outcome"] == 1.0
    expected_q = (1.0 - 0.02) * 0.5 + 0.95 * 0.5  # 0.49 + 0.475 = 0.965
    expected_r = (1.0 + 1.0 + 0.965) / 3.0  # ≈ 0.988
    assert abs(c["r_quality"] - expected_q) < 1e-12
    assert abs(r - expected_r) < 1e-12
    assert r > 0.95


def test_posterior_reward_perfect_oracle() -> None:
    """Perfect inputs (nmse=0, support=1, conformal=1) → R=1.0."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.0, support=1.0, conformal=1.0,
    )
    assert c["r_format"] == 1.0
    assert c["r_outcome"] == 1.0
    assert c["r_quality"] == 1.0
    assert r == 1.0


# ── critical: posterior gating distinction (test 8) ─────────────────


def test_posterior_gating_actually_fires() -> None:
    """**Reviewer-proof test**: when outcome_correct=False, R_quality must
    NOT contribute to the reward, even if R_quality would otherwise be high.

    A naive (ungated) formula would give a much higher reward for the same
    inputs. This test pins the gate's contribution to the math.
    """
    # Inputs: format passes; both outcome conditions fail; quality "would
    # be" high (0.875) under an ungated formula.
    inputs = dict(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.20,   # > 0.10 threshold → outcome_fails
        support=0.40,    # < 0.50 threshold → outcome_fails
        conformal=0.95,
    )
    r, c = posterior_reward(**inputs)

    # What the WOULD-BE quality is, even though the gate kills it.
    would_be_quality = (1.0 - 0.20) * 0.5 + 0.95 * 0.5  # = 0.875
    assert abs(c["r_quality"] - would_be_quality) < 1e-12, (
        "r_quality should still be reported in components for telemetry, "
        "even when the gate zeros its contribution to the reward."
    )

    # Posterior gating: r_outcome=0 → quality contribution is 0.
    # Reward = (R_format + R_outcome + R_outcome × R_quality) / 3
    #        = (1 + 0 + 0 × 0.875) / 3
    #        = 1 / 3
    expected_gated = 1.0 / 3.0
    assert abs(r - expected_gated) < 1e-12, (
        f"posterior reward must be {expected_gated:.6f} (gated), got {r:.6f}"
    )

    # Hypothetical "naive" averaging without the gate would give a much
    # higher reward. Pin the difference to make the gate's effect visible.
    naive_ungated = (1.0 + would_be_quality) / 2.0  # = 0.9375
    assert naive_ungated > 0.5
    assert r < naive_ungated - 0.5, (
        "Posterior reward must be substantially below the naive ungated "
        f"alternative; got posterior={r:.4f} vs naive={naive_ungated:.4f}"
    )


# ── boundary tests (9, 10) ──────────────────────────────────────────


def test_posterior_outcome_boundary_pass() -> None:
    """support=0.5 AND nmse_raw=0.10 exactly → outcome_correct=True."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.10, support=0.5, conformal=0.0,
    )
    assert c["r_outcome"] == 1.0  # ≥ and ≤ are inclusive boundaries


def test_posterior_outcome_boundary_fail_support() -> None:
    """support=0.4999 → outcome_correct=False (strict ≥ on support)."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.10, support=0.4999, conformal=1.0,
    )
    assert c["r_outcome"] == 0.0


def test_posterior_outcome_boundary_fail_nmse() -> None:
    """nmse_raw=0.1001 → outcome_correct=False (strict ≤ on nmse_raw)."""
    r, c = posterior_reward(
        parse_valid=1.0, format_valid=1.0,
        nmse_raw=0.1001, support=1.0, conformal=1.0,
    )
    assert c["r_outcome"] == 0.0


# ── end-to-end tests on the real env (11–14) ────────────────────────


@pytest.fixture(scope="module")
def reward_fn():
    # Posterior-only scope: isolate the gating behaviour from the
    # reasoning-tags layer. End-to-end tag tests live in
    # tests/training/test_reasoning_tags.py.
    return make_reward_fn_posterior(ENV_ID, use_tags=False)


def test_end_to_end_zero_completion_outcome_fails(reward_fn) -> None:
    """Zero-amplitude prediction: format passes, outcome fails → r ≈ 1/3."""
    reward_fn.stats.reset()
    rewards = reward_fn(
        prompts=[""], completions=[_zero_completion()], instance_seed=[0],
    )
    assert len(rewards) == 1
    assert abs(rewards[0] - 1.0 / 3.0) < 1e-9
    rec = reward_fn.stats.per_call[-1]
    assert rec["components"]["r_format"] == 1.0
    assert rec["components"]["r_outcome"] == 0.0


def test_end_to_end_oracle_outcome_passes(reward_fn) -> None:
    """Oracle prediction: outcome passes; reward ≥ 0.95 (typical near 0.98)."""
    reward_fn.stats.reset()
    rewards = reward_fn(
        prompts=[""], completions=[_oracle_completion(0)], instance_seed=[0],
    )
    rec = reward_fn.stats.per_call[-1]
    assert rec["components"]["r_outcome"] == 1.0
    assert rewards[0] > 0.95


def test_end_to_end_determinism_bitwise(reward_fn) -> None:
    """Same (seed, completion) twice → identical reward."""
    reward_fn.stats.reset()
    completion = _oracle_completion(7)
    r1 = reward_fn(prompts=[""], completions=[completion], instance_seed=[7])
    r2 = reward_fn(prompts=[""], completions=[completion], instance_seed=[7])
    assert r1 == r2
    assert repr(r1[0]) == repr(r2[0])


def test_end_to_end_mixed_batch_counters(reward_fn) -> None:
    reward_fn.stats.reset()
    # Each oracle completion is generated for a specific seed; the dataset's
    # instance_seed must match so the reward-fn regenerates the same instance.
    completions = [
        "totally not json",       # parse_fail
        _zero_completion(),       # format_ok, outcome_fail
        _oracle_completion(2),    # format_ok, outcome_pass (matched seed=2)
        _oracle_completion(3),    # format_ok, outcome_pass (matched seed=3)
        _zero_completion(),       # format_ok, outcome_fail
    ]
    rewards = reward_fn(
        prompts=[""] * 5, completions=completions, instance_seed=[0, 1, 2, 3, 4],
    )
    s = reward_fn.stats
    assert s.n_calls == 5
    assert s.n_parse_valid == 4  # all but the garbage
    assert s.n_format_valid == 4  # all but the garbage
    assert s.n_outcome_correct == 2  # only the two seed-aligned oracles
    assert rewards[0] == 0.0  # parse fail
    assert abs(rewards[1] - 1.0 / 3.0) < 1e-9
    assert rewards[2] > 0.95
    assert rewards[3] > 0.95
    assert abs(rewards[4] - 1.0 / 3.0) < 1e-9


def test_end_to_end_50_calls_under_30s(reward_fn) -> None:
    reward_fn.stats.reset()
    n = 50
    completions = [_zero_completion()] * n
    seeds = list(range(n))
    t0 = time.perf_counter()
    rewards = reward_fn(prompts=[""] * n, completions=completions, instance_seed=seeds)
    elapsed = time.perf_counter() - t0
    assert len(rewards) == n
    assert all(math.isfinite(r) for r in rewards)
    assert elapsed < 30.0, f"50 calls took {elapsed:.2f}s, expected < 30s"


# ── stats container ─────────────────────────────────────────────────


def test_posterior_reward_stats_aggregate_keys() -> None:
    s = PosteriorRewardStats()
    s.n_calls = 3
    s.n_parse_valid = 3
    s.n_format_valid = 2
    s.n_outcome_correct = 1
    s.sum_reward = 1.0
    s.sum_quality_when_outcome = 0.8
    agg = s.aggregate()
    expected = {"n_calls", "mean_reward", "parse_valid_rate", "format_valid_rate",
                "outcome_correct_rate", "score_exception_rate",
                "mean_quality_when_outcome"}
    assert set(agg) == expected
    assert agg["outcome_correct_rate"] == pytest.approx(1.0 / 3.0)
    assert agg["mean_quality_when_outcome"] == pytest.approx(0.8)


def test_posterior_components_match_env_meta_nmse_raw() -> None:
    """The wrapper must use score['meta']['nmse_raw'], not the env's
    components['nmse'] (which is the squashed exp(-nmse_raw/τ) value)."""
    fn = make_reward_fn_posterior(ENV_ID, use_tags=False)
    fn(prompts=[""], completions=[_oracle_completion(2)], instance_seed=[2])
    rec = fn.stats.per_call[-1]
    # Oracle has very small nmse_raw — score_nmse (squashed) would be ≈ 1.0
    # which is NEVER ≤ 0.10. If we used the wrong value, outcome_pass would
    # never fire for the oracle.
    assert rec["components"]["nmse_raw"] < 0.05
    assert rec["components"]["r_outcome"] == 1.0
