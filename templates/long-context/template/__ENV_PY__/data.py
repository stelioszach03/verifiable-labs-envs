"""Problem generator for __ENV_ID__.

Each call must produce a fresh problem dict with these keys:

- ``template_name`` — categorical tag for telemetry.
- ``question`` — natural-language question the LLM reads.
- ``corpus`` — the procedurally generated :class:`Corpus`.
- ``needle_text`` — the answer-bearing sentence injected into one
  document.
- ``needle_anchor`` — :class:`NeedleAnchor` (doc id + char offset).
- ``position_mode`` — one of ``"start" | "middle" | "end" | "random"``.

Procedural regeneration from a 64-bit seed plus a finite topic
template pool (10 topics × 4 positions × 1e6 parameter combos) gives
the contamination-resistance guarantee that makes the env safe to
use as an RLVR training signal (EFFECTIVE_INSTANCES > 1e15).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verifiable_labs_envs.long_context_primitives import (
    Corpus,
    NeedleAnchor,
    PositionMode,
)


@dataclass(frozen=True)
class NeedleInstance:
    """One long-context problem draw.

    ``needle_text`` + ``needle_anchor`` are oracle fields excluded
    from :meth:`as_inputs`. ``position_mode`` is exposed so eval
    code can stratify accuracy by needle position.
    """

    question: str
    template_name: str
    seed: int
    corpus: Corpus
    needle_text: str
    needle_anchor: NeedleAnchor
    position_mode: PositionMode
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_answer(self) -> str:
        """The needle text — the canonical answer string."""
        return self.needle_text

    @property
    def prompt(self) -> str:
        """Composed user prompt: documents + question."""
        return self.corpus.render_prompt(question=self.question)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            "context_token_count": self.corpus.total_tokens(),
            "document_count": len(self.corpus.documents),
            "template_name": self.template_name,
            **self.metadata,
        }


@dataclass(frozen=True)
class NeedlePrediction:
    """Solver's answer.

    ``answer`` is the extracted text the model proposes;
    ``raw`` keeps the LLM's full response for traceability;
    ``confidence`` is a scalar self-report in ``[0, 1]``.
    """

    answer: str
    raw: str = ""
    confidence: float = 0.5


def generate_problem(seed: int, **hyperparams: Any) -> dict[str, Any]:
    """Sample a fresh long-context problem from the per-env distribution.

    Determinism: identical ``seed`` returns the byte-identical dict.
    """
    raise NotImplementedError(
        "TODO: implement problem generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility, sample "
        "templates from a pool whose size × seed-space × position "
        "modes × parameter range gives EFFECTIVE_INSTANCES > 1e15. "
        "See src/verifiable_labs_envs/long_context_primitives/__init__.py "
        "for the procedural corpus + needle helpers."
    )


__all__ = ["NeedleInstance", "NeedlePrediction", "generate_problem"]
