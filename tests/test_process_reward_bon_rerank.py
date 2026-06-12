"""Tests for ``verifiable_labs_envs.process_reward.bon_rerank``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.bon_rerank import (
    BonCandidate,
    BonResult,
    bon_lift_metrics,
    make_synthetic_bon_problems,
    passes_bon_lift_floor,
    rerank_bon,
)

# ── synthetic BoN problems ─────────────────────────────────────────


def test_synthetic_bon_basic_shape() -> None:
    problems = make_synthetic_bon_problems(n_problems=5, n_per_problem=3)
    assert len(problems) == 5
    for cands in problems:
        assert len(cands) == 3
        assert all(isinstance(c, BonCandidate) for c in cands)
        for c in cands:
            assert c.env_reward is not None
            assert 0.0 <= c.env_reward <= 1.0


def test_synthetic_bon_deterministic() -> None:
    a = make_synthetic_bon_problems(n_problems=3, n_per_problem=4, seed=42)
    b = make_synthetic_bon_problems(n_problems=3, n_per_problem=4, seed=42)
    a_summary = [(c.prompt, c.env_reward) for cands in a for c in cands]
    b_summary = [(c.prompt, c.env_reward) for cands in b for c in cands]
    assert a_summary == b_summary


def test_synthetic_bon_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="must be"):
        make_synthetic_bon_problems(n_problems=-1)
    with pytest.raises(ValueError, match="must be"):
        make_synthetic_bon_problems(n_per_problem=0)


def test_synthetic_bon_zero_problems() -> None:
    assert make_synthetic_bon_problems(n_problems=0) == []


# ── rerank_bon ─────────────────────────────────────────────────────


def test_rerank_bon_picks_highest_aggregate() -> None:
    cands = [
        BonCandidate(prompt="p", steps=("a",), env_reward=0.3),
        BonCandidate(prompt="p", steps=("b",), env_reward=0.9),
        BonCandidate(prompt="p", steps=("c",), env_reward=0.5),
    ]

    def aggregator(prompt: str, steps) -> float:
        # Score = length of step text; "b" wins.
        return float(len(steps[0]))

    # All three steps are 1 char, so they tie; we'll use a different fn.
    def length_aggregator(prompt: str, steps) -> float:
        # Map step letter to a unique score.
        return {"a": 0.1, "b": 0.9, "c": 0.5}[steps[0]]

    result = rerank_bon(cands, length_aggregator)
    assert isinstance(result, BonResult)
    assert result.chosen_index == 1
    assert result.chosen_aggregate == pytest.approx(0.9)


def test_rerank_bon_tie_breaks_to_lowest_index() -> None:
    cands = [
        BonCandidate(prompt="p", steps=("x",), env_reward=0.5),
        BonCandidate(prompt="p", steps=("y",), env_reward=0.5),
    ]

    def constant(prompt: str, steps) -> float:
        return 0.5

    result = rerank_bon(cands, constant)
    assert result.chosen_index == 0


def test_rerank_bon_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        rerank_bon([], lambda p, s: 0.5)


def test_rerank_bon_propagates_chosen_env_reward() -> None:
    cands = [
        BonCandidate(prompt="p", steps=("a",), env_reward=0.3),
        BonCandidate(prompt="p", steps=("b",), env_reward=0.9),
    ]
    result = rerank_bon(
        cands, lambda p, s: {"a": 0.1, "b": 0.5}[s[0]]
    )
    assert result.chosen_env_reward == pytest.approx(0.9)


# ── bon_lift_metrics ───────────────────────────────────────────────


def test_bon_lift_metrics_perfect_prm_picks_correct() -> None:
    """A PRM that returns env_reward verbatim → always picks the
    correct candidate."""
    problems = make_synthetic_bon_problems(n_problems=10, n_per_problem=4, seed=0)

    def perfect_prm(prompt: str, steps) -> float:
        # Find the candidate this matches and return its env_reward.
        for cands in problems:
            for c in cands:
                if c.steps == steps:
                    return float(c.env_reward or 0.5)
        return 0.5

    metrics = bon_lift_metrics(
        problems, prm_aggregate_predictor=perfect_prm, correct_threshold=0.5
    )
    assert metrics["prm_bon_accuracy"] >= metrics["single_accuracy"]
    assert metrics["n_problems"] == 10


def test_bon_lift_metrics_constant_prm_no_lift() -> None:
    """A constant PRM ties every candidate; reranking falls back to
    index 0 → matches single_accuracy."""
    problems = make_synthetic_bon_problems(n_problems=10, n_per_problem=4, seed=0)
    metrics = bon_lift_metrics(
        problems,
        prm_aggregate_predictor=lambda p, s: 0.5,
        correct_threshold=0.5,
    )
    assert metrics["prm_bon_accuracy"] == pytest.approx(metrics["single_accuracy"])
    assert metrics["prm_bon_lift_vs_single"] == pytest.approx(0.0)


def test_bon_lift_metrics_includes_rm_when_supplied() -> None:
    problems = make_synthetic_bon_problems(n_problems=5, n_per_problem=3, seed=0)
    metrics = bon_lift_metrics(
        problems,
        prm_aggregate_predictor=lambda p, s: 0.5,
        rm_predictor=lambda p, c: 0.5,
        correct_threshold=0.5,
    )
    assert "rm_bon_accuracy" in metrics
    assert "prm_bon_lift_vs_rm" in metrics


def test_bon_lift_metrics_omits_rm_when_not_supplied() -> None:
    problems = make_synthetic_bon_problems(n_problems=3, n_per_problem=2, seed=0)
    metrics = bon_lift_metrics(
        problems,
        prm_aggregate_predictor=lambda p, s: 0.5,
    )
    assert "rm_bon_accuracy" not in metrics
    assert "prm_bon_lift_vs_rm" not in metrics


def test_bon_lift_metrics_empty_input() -> None:
    metrics = bon_lift_metrics(
        [], prm_aggregate_predictor=lambda p, s: 0.5
    )
    assert metrics["n_problems"] == 0


def test_bon_lift_metrics_skips_problems_with_no_env_reward() -> None:
    problems = [[
        BonCandidate(prompt="p", steps=("s",), env_reward=None),
    ]]
    metrics = bon_lift_metrics(
        problems, prm_aggregate_predictor=lambda p, s: 0.5
    )
    # All problems skipped → n_scored=0, single_accuracy=0.
    assert metrics["n_scored"] == 0.0


# ── passes_bon_lift_floor ──────────────────────────────────────────


def test_passes_bon_lift_floor_default() -> None:
    """Plan §5 D6: default floor +5 pp."""
    metrics = {"prm_bon_lift_vs_rm": 0.06}
    assert passes_bon_lift_floor(metrics) is True
    metrics_low = {"prm_bon_lift_vs_rm": 0.04}
    assert passes_bon_lift_floor(metrics_low) is False


def test_passes_bon_lift_floor_returns_false_when_rm_missing() -> None:
    metrics = {"prm_bon_lift_vs_single": 0.10}
    assert passes_bon_lift_floor(metrics) is False


def test_passes_bon_lift_floor_custom_threshold() -> None:
    metrics = {"prm_bon_lift_vs_rm": 0.10}
    assert passes_bon_lift_floor(metrics, floor=0.15) is False
    assert passes_bon_lift_floor(metrics, floor=0.05) is True
