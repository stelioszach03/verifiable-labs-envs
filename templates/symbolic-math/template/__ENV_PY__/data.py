"""Problem generator for __ENV_ID__.

Each call must produce a fresh ``(prompt, gold_expr)`` pair from the
supplied seed. ``prompt`` is the natural-language problem the LLM sees;
``gold_expr`` is the SymPy-parseable canonical form of the correct
answer that the scorer compares against.

Procedural regeneration from a 64-bit seed plus a finite symbolic-pool
gives the contamination-resistance guarantee that makes the env safe to
use as an RLVR training signal. Both ``prompt`` and ``gold_expr`` are
stored as **strings**, not ``sympy.Expr`` objects — keeps instances
pickleable and avoids importing SymPy at module load time.

TODO: replace the body of :func:`generate_problem` with your domain's
generator. Examples:

- ``math-algebra`` (single-turn): random binomial product to expand,
  random sum-of-monomials to factor, random polynomial equality to
  verify.
- ``math-algebra-multiturn``: same generator, multi-turn dialogue.
- ``math-algebra-tools``: same generator, model composes
  ``simplify``/``expand``/``solve``/``substitute`` primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Instance:
    """One problem draw.

    ``gold_expr`` is the oracle field used by the scorer only — solvers
    must treat it as hidden (excluded by :meth:`as_inputs`).
    """

    prompt: str
    gold_expr: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "prompt": self.prompt,
            **self.metadata,
        }


@dataclass(frozen=True)
class Prediction:
    """Solver's answer.

    ``answer_expr`` carries the SymPy-parseable answer string;
    ``confidence`` is a scalar self-report in ``[0, 1]`` used by the
    conformal layer (residual-quantile threshold). ``raw`` keeps the
    LLM's full text response for the audit trail.
    """

    answer_expr: str
    raw: str = ""
    confidence: float = 0.5


def generate_problem(seed: int, **hyperparams: Any) -> tuple[str, str]:
    """Sample a fresh ``(prompt, gold_expr)`` pair from the per-env distribution.

    Determinism: two calls with the same seed and hyperparameters must
    return identical ``(prompt, gold_expr)`` tuples.

    Returns
    -------
    (prompt, gold_expr) : tuple[str, str]
        ``prompt`` is the LLM-facing natural-language problem;
        ``gold_expr`` is the SymPy-parseable canonical answer.
    """
    raise NotImplementedError(
        "TODO: implement problem generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility, sample "
        "symbolic templates from a pool whose size × seed-space gives "
        "EFFECTIVE_INSTANCES > 1e15."
    )


__all__ = ["Instance", "Prediction", "generate_problem"]
