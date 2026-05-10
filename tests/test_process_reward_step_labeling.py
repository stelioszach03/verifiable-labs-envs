"""Tests for ``verifiable_labs_envs.process_reward.step_labeling``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.step_labeling import (
    DEFAULT_FALLBACK_DECOMPOSITION,
    StepLabelOutcome,
    env_supports_procedural_decomposition,
    label_steps,
    step_label_summary,
)

# ── decomposition routing ──────────────────────────────────────────


def test_text_env_uses_text_progress_decomposition() -> None:
    out = label_steps(
        env_id="math-algebra",
        steps=["Step 1.", "Step 2.", "Step 3."],
        outcome_reward=1.0,
    )
    assert out.decomposition == "text_progress"
    assert out.step_count == 3


def test_non_text_env_uses_terminal_uniform() -> None:
    out = label_steps(
        env_id="sparse-fourier-recovery",
        steps=["s1", "s2"],
        outcome_reward=0.7,
    )
    assert out.decomposition == DEFAULT_FALLBACK_DECOMPOSITION
    assert out.step_rewards == (pytest.approx(0.7), pytest.approx(0.7))


def test_unknown_env_falls_back() -> None:
    out = label_steps(
        env_id="not-a-real-env",
        steps=["s1"],
        outcome_reward=0.5,
    )
    assert out.decomposition == DEFAULT_FALLBACK_DECOMPOSITION


def test_none_env_falls_back() -> None:
    out = label_steps(
        env_id=None, steps=["s1"], outcome_reward=0.5
    )
    assert out.decomposition == DEFAULT_FALLBACK_DECOMPOSITION


# ── text-env progress monotonicity ─────────────────────────────────


def test_text_env_terminal_step_locks_at_outcome() -> None:
    """Plan §5 D2-D: final step inherits outcome reward."""
    out = label_steps(
        env_id="math-algebra",
        steps=["A", "B", "C"],
        outcome_reward=0.9,
    )
    assert out.step_rewards[-1] == pytest.approx(0.9)


def test_text_env_progress_monotonically_increases() -> None:
    out = label_steps(
        env_id="sql-multiturn",
        steps=["A", "B", "C", "D"],
        outcome_reward=0.8,
    )
    # Earlier steps have lower progress credit.
    assert out.step_rewards[0] < out.step_rewards[-1]


def test_text_env_clipped_to_unit() -> None:
    """Outcome > 1.0 → clipped."""
    out = label_steps(
        env_id="math-algebra",
        steps=["A", "B"],
        outcome_reward=1.5,
    )
    for r in out.step_rewards:
        assert 0.0 <= r <= 1.0


def test_text_env_clipped_at_zero() -> None:
    out = label_steps(
        env_id="math-algebra",
        steps=["A"],
        outcome_reward=-0.5,
    )
    assert out.step_rewards[0] == 0.0


def test_empty_steps_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        label_steps(env_id="math-algebra", steps=[], outcome_reward=0.5)


# ── components passthrough ─────────────────────────────────────────


def test_text_env_components_carry_parse_credit() -> None:
    out = label_steps(
        env_id="math-algebra",
        steps=["A", "B"],
        outcome_reward=0.5,
        components={"parse_valid": 0.8},
    )
    for comp in out.step_components:
        assert comp is not None
        assert "parse_valid" in comp


def test_terminal_uniform_carries_components_dict() -> None:
    """Fallback returns the input components for every step."""
    out = label_steps(
        env_id="sparse-fourier-recovery",
        steps=["s1", "s2"],
        outcome_reward=0.5,
        components={"some_metric": 0.42},
    )
    for comp in out.step_components:
        assert comp == {"some_metric": 0.42}


# ── parse-credit heuristic edge cases ──────────────────────────────


def test_empty_step_text_gets_zero_parse_credit() -> None:
    """An empty step contributes zero parse_credit (heuristic)."""
    out = label_steps(
        env_id="math-algebra",
        steps=["", "B"],
        outcome_reward=1.0,
    )
    # First step has parse_credit=0 → reward=0.
    assert out.step_rewards[0] == pytest.approx(0.0)


def test_punctuation_only_step_gets_partial_parse_credit() -> None:
    """A punctuation-only step gets 0.5 parse_credit."""
    out = label_steps(
        env_id="math-algebra",
        steps=[".....", "Real step"],
        outcome_reward=1.0,
    )
    assert out.step_rewards[0] < out.step_rewards[1]


# ── env_supports_procedural_decomposition predicate ────────────────


def test_predicate_recognises_text_envs() -> None:
    assert env_supports_procedural_decomposition("math-algebra") is True
    assert env_supports_procedural_decomposition("sql-multiturn") is True
    assert env_supports_procedural_decomposition("code-mini-repo") is True
    assert env_supports_procedural_decomposition("long-context-synthesis") is True


def test_predicate_rejects_array_envs() -> None:
    assert env_supports_procedural_decomposition("sparse-fourier-recovery") is False
    assert env_supports_procedural_decomposition("phase-retrieval") is False
    assert env_supports_procedural_decomposition(None) is False


# ── summary ────────────────────────────────────────────────────────


def test_step_label_summary_basic() -> None:
    out = label_steps(
        env_id="math-algebra",
        steps=["A", "B", "C"],
        outcome_reward=0.6,
    )
    summary = step_label_summary(out)
    assert summary["n_steps"] == 3
    assert 0.0 <= summary["min"] <= summary["mean"] <= summary["max"] <= 1.0


def test_step_label_summary_empty() -> None:
    empty = StepLabelOutcome(
        step_rewards=(),
        step_components=(),
        decomposition="terminal_uniform",
    )
    summary = step_label_summary(empty)
    assert summary["n_steps"] == 0
    assert summary["mean"] == 0.0


# ── determinism ────────────────────────────────────────────────────


def test_label_steps_deterministic() -> None:
    a = label_steps(
        env_id="math-algebra", steps=["A", "B"], outcome_reward=0.7
    )
    b = label_steps(
        env_id="math-algebra", steps=["A", "B"], outcome_reward=0.7
    )
    assert a == b
