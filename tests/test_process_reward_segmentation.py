"""Tests for ``verifiable_labs_envs.process_reward.segmentation``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.segmentation import (
    DEFAULT_MAX_STEPS,
    LOW_CONFIDENCE_THRESHOLD,
    SegmentationOutcome,
    is_pre_segmented,
    segment_trace,
    steps_to_trace,
)

# ── default_max_steps locked ────────────────────────────────────────


def test_default_max_steps_locked_at_32() -> None:
    """Plan §17 invariant 11: max_steps cap = 32 (R15)."""
    assert DEFAULT_MAX_STEPS == 32


def test_low_confidence_threshold_locked() -> None:
    assert LOW_CONFIDENCE_THRESHOLD == 0.5


# ── pre-segmented passthrough ───────────────────────────────────────


def test_pre_segmented_passthrough() -> None:
    out = segment_trace(["First step.", "Second step.", "Third step."])
    assert out.steps == ("First step.", "Second step.", "Third step.")
    assert out.strategy == "pre_segmented"
    assert out.confidence == pytest.approx(1.0)
    assert out.warning is None
    assert out.truncated is False


def test_pre_segmented_strips_each_entry() -> None:
    out = segment_trace(["  First.  ", "Second."])
    assert out.steps == ("First.", "Second.")


def test_pre_segmented_drops_empty_entries() -> None:
    out = segment_trace(["First.", "", "Second."])
    # Empty strings drop; result is ("First.", "Second.").
    assert out.steps == ("First.", "Second.")


def test_pre_segmented_all_empty_yields_single_empty_step() -> None:
    out = segment_trace([])
    # Empty list → single empty step (loop-safety contract).
    assert out.step_count == 1
    assert out.steps == ("",)


# ── D14-B explicit Step N: marker ───────────────────────────────────


def test_explicit_step_marker_basic() -> None:
    raw = "Step 1: Do A.\nStep 2: Do B.\nStep 3: Do C."
    out = segment_trace(raw)
    assert out.strategy == "explicit_step_marker"
    assert out.confidence == pytest.approx(0.95)
    assert out.steps == ("Do A.", "Do B.", "Do C.")
    assert out.warning is None


def test_explicit_step_marker_case_insensitive() -> None:
    raw = "STEP 1: A\nstep 2. B\nStep 3) C"
    out = segment_trace(raw)
    assert out.strategy == "explicit_step_marker"
    assert len(out.steps) == 3


def test_explicit_step_marker_with_preamble() -> None:
    raw = "Let me solve this.\nStep 1: A\nStep 2: B"
    out = segment_trace(raw)
    assert out.strategy == "explicit_step_marker"
    # Preamble preserved as the first step.
    assert out.steps[0] == "Let me solve this."
    assert "A" in out.steps[1]
    assert "B" in out.steps[2]


# ── D14-A numbered lines ────────────────────────────────────────────


def test_numbered_lines() -> None:
    raw = "1. Do A.\n2. Do B.\n3. Do C."
    out = segment_trace(raw)
    assert out.strategy == "numbered_ordinal"
    assert out.confidence == pytest.approx(0.85)
    assert len(out.steps) == 3


def test_numbered_lines_with_preamble() -> None:
    raw = "Plan:\n1. A\n2. B"
    out = segment_trace(raw)
    assert out.strategy == "numbered_ordinal"
    assert out.steps[0] == "Plan:"


# ── D14-A ordinal cues ──────────────────────────────────────────────


def test_ordinal_cues() -> None:
    raw = "First, do A.\nSecond, do B.\nFinally, do C."
    out = segment_trace(raw)
    assert out.strategy == "numbered_ordinal"
    # Confidence is 0.75 for the ordinal cue path.
    assert out.confidence == pytest.approx(0.75)
    assert len(out.steps) == 3


# ── paragraph-break fallback ────────────────────────────────────────


def test_paragraph_break_fallback() -> None:
    """Use non-ordinal-prefixed paragraphs so we land at newline_only,
    not the higher-priority ordinal-cue path."""
    raw = "Initial chunk of text.\n\nMiddle chunk of text.\n\nFinal chunk of text."
    out = segment_trace(raw)
    assert out.strategy == "newline_only"
    assert out.confidence == pytest.approx(0.6)
    assert len(out.steps) == 3


# ── D14-C sentence-boundary fallback ────────────────────────────────


def test_sentence_boundary_fallback() -> None:
    raw = "I think x is five. Then y is ten. So z is fifteen."
    out = segment_trace(raw)
    assert out.strategy == "sentence_boundary"
    assert out.confidence == pytest.approx(0.4)
    assert out.warning == "low_confidence"
    assert len(out.steps) == 3


def test_sentence_boundary_is_low_confidence() -> None:
    raw = "I think x is five. Then y is ten."
    out = segment_trace(raw)
    assert out.is_low_confidence() is True


# ── single-step fallback ────────────────────────────────────────────


def test_single_step_fallback_for_unstructured() -> None:
    """A single short sentence with no boundaries → single_step."""
    out = segment_trace("just one short answer")
    assert out.strategy == "single_step"
    assert out.steps == ("just one short answer",)
    assert out.warning == "low_confidence"


def test_empty_trace_yields_single_empty_step() -> None:
    out = segment_trace("")
    assert out.steps == ("",)
    assert out.is_low_confidence()


def test_whitespace_only_trace_yields_single_empty_step() -> None:
    out = segment_trace("   \n\n  ")
    assert out.steps == ("",)


# ── truncation ──────────────────────────────────────────────────────


def test_truncation_to_max_steps() -> None:
    raw = "\n".join(f"Step {i}: x" for i in range(1, 50))
    out = segment_trace(raw, max_steps=5)
    assert out.step_count == 5
    assert out.truncated is True
    assert out.warning in ("truncated", "low_confidence")


def test_no_truncation_under_cap() -> None:
    raw = "Step 1: A\nStep 2: B"
    out = segment_trace(raw, max_steps=10)
    assert out.truncated is False


def test_max_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        segment_trace("anything", max_steps=0)


# ── determinism (R10 invariant) ─────────────────────────────────────


def test_segmenter_is_deterministic() -> None:
    raw = "Step 1: do this.\nStep 2: do that."
    a = segment_trace(raw)
    b = segment_trace(raw)
    assert a.steps == b.steps
    assert a.strategy == b.strategy
    assert a.confidence == b.confidence


def test_segmenter_pre_segmented_is_deterministic() -> None:
    a = segment_trace(["one", "two"])
    b = segment_trace(["one", "two"])
    assert a == b


# ── helpers ─────────────────────────────────────────────────────────


def test_steps_to_trace_round_trip() -> None:
    steps = ["First.", "Second.", "Third."]
    joined = steps_to_trace(steps)
    assert joined == "First.\nSecond.\nThird."


def test_is_pre_segmented_predicate() -> None:
    assert is_pre_segmented(["a", "b"]) is True
    assert is_pre_segmented("a\nb") is False


def test_outcome_dataclass_fields() -> None:
    out = SegmentationOutcome(
        steps=("a", "b"),
        strategy="pre_segmented",
        confidence=1.0,
    )
    assert out.step_count == 2
    assert out.is_low_confidence() is False


def test_min_step_chars_filter() -> None:
    """min_step_chars drops short steps."""
    out = segment_trace(["short", "longer step", "x"], min_step_chars=4)
    # Only "short" (5) and "longer step" (11) survive; "x" is dropped.
    assert out.steps == ("short", "longer step")
