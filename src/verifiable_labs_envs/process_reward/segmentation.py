"""D14-D hybrid step segmentation for process-reward traces.

Per :doc:`PHASE_30_PLAN.md` §5 D14-D + §17 hard guard 10:

- **D14-A** — numbered marker primary (``"\\n"`` separates steps when
  preceded by an ordinal cue: ``"First, ...\\nSecond, ..."``).
- **D14-B** — explicit ``Step N:`` regex high-confidence override
  (matches ``"Step 1:"``, ``"step 2)"``, ``"STEP 3 -"`` etc.).
- **D14-C** — sentence-boundary fallback for unstructured traces
  (pure-Python; no NLTK/spaCy dependency in the no-GPU path).
- **Customer-supplied** — when the customer POSTs
  ``reasoning_trace: list[str]`` the segmenter passes through with
  full confidence and no warning.

The segmenter is **deterministic** (no `random`, no seeded RNG): same
input produces bit-identical output across runs (R10 invariant).

Boundary confidence is in ``[0, 1]``; the API surface flags
``segmentation_warning="low_confidence"`` when the highest-priority
match for the trace is < 0.5, so the customer can re-segment
client-side.

Hard cap: ``DEFAULT_MAX_STEPS = 32`` per R15 (training data tail
distribution); traces longer than the cap are truncated and the
outcome metadata records ``truncated=True``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

DEFAULT_MAX_STEPS: int = 32
"""Per :doc:`PHASE_30_PLAN.md` R15 — truncation cap on step count.
Configurable via :func:`segment_trace(..., max_steps=N)`."""

LOW_CONFIDENCE_THRESHOLD: float = 0.5
"""Below this the segmenter emits ``low_confidence`` warning."""

SegmentationStrategy = Literal[
    "explicit_step_marker",     # D14-B "Step N:" regex hit
    "numbered_ordinal",          # D14-A "First/Second/..." or "1.\n2.\n"
    "newline_only",              # split on bare \n\n paragraph breaks
    "sentence_boundary",         # D14-C fallback
    "single_step",               # whole trace = 1 step (low confidence)
    "pre_segmented",             # customer supplied list[str]; passthrough
]


@dataclass(frozen=True)
class SegmentationOutcome:
    """One segmentation result + metadata.

    ``steps`` is the post-segmentation step list (always non-empty;
    even an empty trace becomes a single empty step). ``confidence``
    is in ``[0, 1]``; ``warning`` is ``"low_confidence"`` when the
    confidence is below :data:`LOW_CONFIDENCE_THRESHOLD` and ``None``
    otherwise. ``strategy`` records which D14 path was selected so
    the audit trail captures it.
    """

    steps: tuple[str, ...]
    strategy: SegmentationStrategy
    confidence: float
    truncated: bool = False
    warning: str | None = None
    raw_trace: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def is_low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD


# ── compiled regex helpers ──────────────────────────────────────────


_STEP_MARKER_RE = re.compile(
    r"(?im)^\s*step\s*(\d+)\s*[:.\)\-]\s*",
)
"""Matches ``Step 1:``, ``step 2.``, ``Step 3)``, ``STEP 4 -`` etc.
Multiline + case-insensitive."""

_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*("
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"finally|next|then|now"
    r")[,.\s]",
    re.IGNORECASE,
)
"""Matches ``First, ``, ``Second.``, ``Then `` at the head of a line —
the D14-A heuristic."""

_NUMBERED_LINE_RE = re.compile(
    r"^\s*(\d+)\s*[.)\-:]\s+",
)
"""Matches ``1. ``, ``2)``, ``3-``, ``4: `` at the head of a line."""

_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])",
)
"""Pure-Python sentence boundary heuristic: punctuation followed by
whitespace and a capital letter. Mirrors a coarse NLTK punkt
behaviour without the dependency."""


# ── entry points ────────────────────────────────────────────────────


def segment_trace(
    trace: str | Sequence[str],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    min_step_chars: int = 1,
) -> SegmentationOutcome:
    """Segment a reasoning trace into discrete steps.

    Behaviour ordered by D14-D priority:

    1. **Pre-segmented:** ``trace`` is a sequence (not a single string);
       passthrough at confidence 1.0.
    2. **Explicit ``Step N:`` markers (D14-B):** confidence 0.95.
    3. **Numbered lines (``1.``/``2)``) (D14-A):** confidence 0.85
       when ≥ 2 sequentially-numbered lines are present.
    4. **Ordinal cues (``First, .../Second, ...``) (D14-A):**
       confidence 0.75.
    5. **Bare paragraph breaks (``\\n\\n``):** confidence 0.6.
    6. **Sentence boundary (D14-C):** confidence 0.4 (low; emits
       warning).
    7. **Whole trace as 1 step:** confidence 0.3 (warning).

    ``max_steps`` truncates the result; the outcome's ``truncated``
    flag is ``True`` when truncation occurred. ``min_step_chars``
    drops candidate steps shorter than the threshold (default 1; some
    callers raise this to filter out blank lines).
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive; got {max_steps}")
    if min_step_chars < 0:
        raise ValueError(f"min_step_chars must be non-negative; got {min_step_chars}")

    if not isinstance(trace, str):
        return _from_pre_segmented(trace, max_steps=max_steps, min_step_chars=min_step_chars)

    raw = trace
    if not raw.strip():
        return SegmentationOutcome(
            steps=("",),
            strategy="single_step",
            confidence=0.3,
            truncated=False,
            warning="low_confidence",
            raw_trace=raw,
        )

    for builder in (
        _try_explicit_step_marker,
        _try_numbered_lines,
        _try_ordinal_cues,
        _try_paragraph_breaks,
        _try_sentence_boundary,
    ):
        outcome = builder(raw, min_step_chars=min_step_chars)
        if outcome is not None:
            return _truncate(outcome, max_steps=max_steps)

    # Last resort — whole trace as 1 step.
    return SegmentationOutcome(
        steps=(raw.strip(),),
        strategy="single_step",
        confidence=0.3,
        truncated=False,
        warning="low_confidence",
        raw_trace=raw,
    )


def _from_pre_segmented(
    trace: Sequence[str], *, max_steps: int, min_step_chars: int
) -> SegmentationOutcome:
    cleaned = [s.strip() for s in trace if isinstance(s, str)]
    cleaned = [s for s in cleaned if len(s) >= min_step_chars]
    if not cleaned:
        cleaned = [""]
    return _truncate(
        SegmentationOutcome(
            steps=tuple(cleaned),
            strategy="pre_segmented",
            confidence=1.0,
            truncated=False,
            raw_trace="\n".join(cleaned),
        ),
        max_steps=max_steps,
    )


def _try_explicit_step_marker(
    raw: str, *, min_step_chars: int
) -> SegmentationOutcome | None:
    """D14-B path: split on explicit ``Step N:`` markers."""
    matches = list(_STEP_MARKER_RE.finditer(raw))
    if len(matches) < 1:
        return None
    if len(matches) == 1 and matches[0].start() == 0 and "\n" not in raw:
        # Single "Step 1: ..." with no further structure → unhelpful.
        return None
    pieces: list[str] = []
    last_end = 0
    if matches[0].start() > 0:
        head = raw[: matches[0].start()].strip()
        if head and len(head) >= min_step_chars:
            pieces.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[m.end() : end].strip()
        if chunk and len(chunk) >= min_step_chars:
            pieces.append(chunk)
        last_end = end
    del last_end  # not used; documented for the dead-store reader
    if not pieces:
        return None
    return SegmentationOutcome(
        steps=tuple(pieces),
        strategy="explicit_step_marker",
        confidence=0.95,
        raw_trace=raw,
        metadata={"match_count": len(matches)},
    )


def _try_numbered_lines(
    raw: str, *, min_step_chars: int
) -> SegmentationOutcome | None:
    """D14-A path: lines starting with ``1.`` / ``2)`` / ``3 -``."""
    lines = raw.splitlines()
    starts: list[int] = []
    for i, line in enumerate(lines):
        if _NUMBERED_LINE_RE.match(line):
            starts.append(i)
    if len(starts) < 2:
        return None
    pieces: list[str] = []
    if starts[0] > 0:
        head = "\n".join(lines[: starts[0]]).strip()
        if head and len(head) >= min_step_chars:
            pieces.append(head)
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        body_no_marker = _NUMBERED_LINE_RE.sub("", body, count=1).strip()
        if body_no_marker and len(body_no_marker) >= min_step_chars:
            pieces.append(body_no_marker)
    if not pieces:
        return None
    return SegmentationOutcome(
        steps=tuple(pieces),
        strategy="numbered_ordinal",
        confidence=0.85,
        raw_trace=raw,
        metadata={"marker_count": len(starts)},
    )


def _try_ordinal_cues(
    raw: str, *, min_step_chars: int
) -> SegmentationOutcome | None:
    """D14-A weaker path: ``First, ... Second, ...`` ordinal-prefixed lines."""
    lines = raw.splitlines()
    starts: list[int] = []
    for i, line in enumerate(lines):
        if _ORDINAL_PREFIX_RE.match(line):
            starts.append(i)
    if len(starts) < 2:
        return None
    pieces: list[str] = []
    if starts[0] > 0:
        head = "\n".join(lines[: starts[0]]).strip()
        if head and len(head) >= min_step_chars:
            pieces.append(head)
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if body and len(body) >= min_step_chars:
            pieces.append(body)
    if not pieces:
        return None
    return SegmentationOutcome(
        steps=tuple(pieces),
        strategy="numbered_ordinal",
        confidence=0.75,
        raw_trace=raw,
        metadata={"ordinal_count": len(starts)},
    )


def _try_paragraph_breaks(
    raw: str, *, min_step_chars: int
) -> SegmentationOutcome | None:
    """Bare ``\\n\\n`` paragraph splits — moderate confidence."""
    pieces = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    pieces = [p for p in pieces if len(p) >= min_step_chars]
    if len(pieces) < 2:
        return None
    return SegmentationOutcome(
        steps=tuple(pieces),
        strategy="newline_only",
        confidence=0.6,
        raw_trace=raw,
        metadata={"paragraph_count": len(pieces)},
    )


def _try_sentence_boundary(
    raw: str, *, min_step_chars: int
) -> SegmentationOutcome | None:
    """D14-C sentence-boundary fallback."""
    pieces = [
        p.strip() for p in _SENTENCE_BOUNDARY_RE.split(raw) if p.strip()
    ]
    pieces = [p for p in pieces if len(p) >= min_step_chars]
    if len(pieces) < 2:
        return None
    return SegmentationOutcome(
        steps=tuple(pieces),
        strategy="sentence_boundary",
        confidence=0.4,
        raw_trace=raw,
        warning="low_confidence",
        metadata={"sentence_count": len(pieces)},
    )


def _truncate(
    outcome: SegmentationOutcome, *, max_steps: int
) -> SegmentationOutcome:
    if len(outcome.steps) <= max_steps:
        return outcome
    return SegmentationOutcome(
        steps=outcome.steps[:max_steps],
        strategy=outcome.strategy,
        confidence=outcome.confidence,
        truncated=True,
        warning=outcome.warning or "truncated",
        raw_trace=outcome.raw_trace,
        metadata={**outcome.metadata, "original_step_count": len(outcome.steps)},
    )


def steps_to_trace(steps: Iterable[str], separator: str = "\n") -> str:
    """Inverse of :func:`segment_trace` for a pre-segmented array
    (lossy for the other strategies because punctuation separators
    are dropped). Used by :mod:`vlabs_prm_data` CLI to reconstruct
    a flat string for hashing."""
    return separator.join(s.strip() for s in steps if s)


def is_pre_segmented(trace: str | Sequence[str]) -> bool:
    """Predicate: did the customer POST a pre-segmented array?"""
    return not isinstance(trace, str)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "LOW_CONFIDENCE_THRESHOLD",
    "SegmentationOutcome",
    "SegmentationStrategy",
    "is_pre_segmented",
    "segment_trace",
    "steps_to_trace",
]
