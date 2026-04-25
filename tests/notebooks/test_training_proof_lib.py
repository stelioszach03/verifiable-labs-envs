"""Unit tests for ``notebooks/training_proof_lib.py``.

We exercise the full ``evaluate_prompt`` path against the real
multi-turn sparse-Fourier env, but with a ``FakeLLMSolver`` that knows
the ground truth — that lets us verify the wrapper, budget tracking,
parse-failure recording, and bootstrap CI without spending OpenRouter
credits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import training_proof_lib as lib  # noqa: E402

from verifiable_labs_envs import load_environment  # noqa: E402
from verifiable_labs_envs.solvers import FakeLLMSolver  # noqa: E402


def _truth_responder(usd_cost: float = 0.001):
    """Fake solver factory: replies with the ground-truth answer for whatever
    instance the env passed in. We piggyback on ``messages`` to extract n/k
    from the user prompt — but that's fragile; instead we close over the
    instance via a global the test sets.

    The simplest reliable approach: have the test call ``set_instance`` with
    the current instance, and the responder reads it. We just create one
    fake per evaluate_prompt call.
    """
    raise RuntimeError("use _make_truth_solver below")


def _make_truth_solver(instance, usd_cost: float = 0.001) -> FakeLLMSolver:
    """Solver that always returns the truth for ``instance``."""
    truth = json.dumps({
        "support_idx": [int(i) for i in instance.support_true],
        "support_amp_x1000": [
            int(round(float(v) * 1000))
            for v in instance.x_true[instance.support_true]
        ],
    })
    return FakeLLMSolver(response=truth, usd_cost=usd_cost)


def _make_zero_solver(instance, usd_cost: float = 0.001) -> FakeLLMSolver:
    """Solver that returns all-zero amplitudes (a valid but bad answer)."""
    payload = json.dumps({
        "support_idx": [int(i) for i in instance.support_true],
        "support_amp_x1000": [0] * len(instance.support_true),
    })
    return FakeLLMSolver(response=payload, usd_cost=usd_cost)


# ── Budget cap ──────────────────────────────────────────


def test_budget_cap_aborts_when_exceeded():
    cap = lib.BudgetCap(cap_usd=0.10)
    cap.add(0.05)
    cap.add(0.04)
    assert cap.spent_usd == pytest.approx(0.09)
    cap.add(0.005)  # still under
    with pytest.raises(RuntimeError, match="Budget cap exceeded"):
        cap.add(0.10)


def test_budget_cap_ignores_none_cost():
    cap = lib.BudgetCap(cap_usd=1.0)
    cap.add(None)
    cap.add(None)
    assert cap.spent_usd == 0.0
    assert cap.remaining_usd == 1.0


# ── evaluate_prompt with truth-fed fake ───────────────


def test_evaluate_prompt_truth_solver_scores_high():
    env = load_environment("sparse-fourier-recovery-multiturn", calibration_quantile=2.0)
    seeds = [0, 1, 2]
    candidate = lib.PromptCandidate(name="truth", system_prompt="ignored")
    budget = lib.BudgetCap(cap_usd=1.0)

    # The fake produces a different truth response per seed — we build it
    # by chaining a list, one entry per turn (3) per seed.
    canned: list[str] = []
    for s in seeds:
        inst = env.generate_instance(seed=s)
        truth = json.dumps({
            "support_idx": [int(i) for i in inst.support_true],
            "support_amp_x1000": [
                int(round(float(v) * 1000))
                for v in inst.x_true[inst.support_true]
            ],
        })
        canned.extend([truth] * env.max_turns)
    fake = FakeLLMSolver(response=canned, usd_cost=0.0005)

    results = lib.evaluate_prompt(env, fake, candidate, seeds, budget)
    assert len(results) == len(seeds)
    assert all(r.parse_ok for r in results)
    # Truth in, high reward out.
    assert all(r.reward > 0.85 for r in results), [r.reward for r in results]
    assert budget.spent_usd > 0  # accounted


def test_evaluate_prompt_records_parse_failures():
    env = load_environment("sparse-fourier-recovery-multiturn", calibration_quantile=2.0)
    candidate = lib.PromptCandidate(name="garbage", system_prompt="ignored")
    budget = lib.BudgetCap(cap_usd=1.0)

    # Solver always emits unparseable text — both turns fail, so the
    # rollout raises LLMSolverError which evaluate_prompt should catch.
    fake = FakeLLMSolver(response="not json at all", usd_cost=0.0005)
    results = lib.evaluate_prompt(env, fake, candidate, [0], budget)
    assert len(results) == 1
    assert results[0].parse_ok is False
    assert results[0].reward == 0.0


def test_evaluate_prompt_threads_system_override():
    """The candidate's system_prompt must replace the adapter's default
    in the messages the wrapped solver sees."""
    env = load_environment("sparse-fourier-recovery-multiturn", calibration_quantile=2.0)
    inst = env.generate_instance(seed=7)
    truth = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [
            int(round(float(v) * 1000))
            for v in inst.x_true[inst.support_true]
        ],
    })
    fake = FakeLLMSolver(response=[truth, truth, truth], usd_cost=0.0)
    candidate = lib.PromptCandidate(name="overridden", system_prompt="MY-OVERRIDE-TOKEN")
    budget = lib.BudgetCap(cap_usd=1.0)
    lib.evaluate_prompt(env, fake, candidate, [7], budget)

    # Inspect what the fake actually saw.
    seen_system_messages = [
        msgs[0]["content"] for msgs in fake.turn_calls if msgs and msgs[0]["role"] == "system"
    ]
    assert seen_system_messages, "no system messages observed"
    assert all("MY-OVERRIDE-TOKEN" in s for s in seen_system_messages)


def test_evaluate_prompt_budget_exceeded_raises_partway():
    env = load_environment("sparse-fourier-recovery-multiturn", calibration_quantile=2.0)
    candidate = lib.PromptCandidate(name="costly", system_prompt="ignored")
    budget = lib.BudgetCap(cap_usd=0.001)
    fake = FakeLLMSolver(response="not parseable", usd_cost=0.005)  # one call > cap
    with pytest.raises(RuntimeError, match="Budget cap exceeded"):
        lib.evaluate_prompt(env, fake, candidate, [0, 1], budget)


# ── statistics helpers ────────────────────────────────


def test_paired_bootstrap_detects_real_difference():
    a = [0.4, 0.5, 0.45, 0.42, 0.5]
    b = [bi + 0.15 for bi in a]
    out = lib.paired_bootstrap_ci(a, b, n_bootstrap=2000, seed=0)
    assert out.delta == pytest.approx(0.15)
    assert out.lo > 0.0  # CI strictly above zero
    assert out.n == 5


def test_paired_bootstrap_no_difference_includes_zero():
    a = [0.5, 0.6, 0.55, 0.5, 0.62]
    b = [0.51, 0.6, 0.54, 0.5, 0.62]
    out = lib.paired_bootstrap_ci(a, b, n_bootstrap=2000, seed=0)
    assert abs(out.delta) < 0.02
    assert out.lo <= 0.0 <= out.hi


def test_paired_bootstrap_validates_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        lib.paired_bootstrap_ci([1.0, 2.0], [3.0])
    with pytest.raises(ValueError, match="at least one"):
        lib.paired_bootstrap_ci([], [])


def test_summarise_aggregates_correctly():
    rs = [
        lib.SeedResult(seed=0, prompt_name="A", reward=0.6, n_turns=3, parse_ok=True, usd_cost=0.001),
        lib.SeedResult(seed=1, prompt_name="A", reward=0.4, n_turns=3, parse_ok=True, usd_cost=0.001),
        lib.SeedResult(seed=2, prompt_name="A", reward=0.0, n_turns=0, parse_ok=False, usd_cost=0.001),
    ]
    s = lib.summarise(rs)
    assert s.prompt_name == "A"
    assert s.n == 3
    assert s.mean == pytest.approx((0.6 + 0.4 + 0.0) / 3)
    assert s.parse_fail_rate == pytest.approx(1 / 3)


def test_summarise_rejects_mixed_prompt_names():
    rs = [
        lib.SeedResult(seed=0, prompt_name="A", reward=0.5, n_turns=3, parse_ok=True, usd_cost=0.0),
        lib.SeedResult(seed=1, prompt_name="B", reward=0.5, n_turns=3, parse_ok=True, usd_cost=0.0),
    ]
    with pytest.raises(ValueError, match="one prompt"):
        lib.summarise(rs)


def test_best_candidate_picks_max_mean_with_parse_tiebreak():
    summaries = [
        lib.RewardSummary(prompt_name="A", n=5, mean=0.50, std=0.0, parse_fail_rate=0.0),
        lib.RewardSummary(prompt_name="B", n=5, mean=0.50, std=0.0, parse_fail_rate=0.20),
        lib.RewardSummary(prompt_name="C", n=5, mean=0.55, std=0.0, parse_fail_rate=0.10),
    ]
    assert lib.best_candidate(summaries).prompt_name == "C"
    # If we drop C, A should win the tie-break (lower parse-fail rate).
    summaries.pop()
    assert lib.best_candidate(summaries).prompt_name == "A"


# ── DEFAULT_CANDIDATES ────────────────────────────────


def test_default_candidates_are_distinct_and_well_formed():
    candidates = lib.DEFAULT_CANDIDATES
    assert len(candidates) >= 3
    names = [c.name for c in candidates]
    assert len(set(names)) == len(names), "duplicate candidate names"
    for c in candidates:
        assert c.system_prompt.strip(), f"candidate {c.name} has empty prompt"
        assert "support_idx" in c.system_prompt, f"candidate {c.name} forgot the schema"
        assert "support_amp_x1000" in c.system_prompt, (
            f"candidate {c.name} forgot the amplitude key"
        )
