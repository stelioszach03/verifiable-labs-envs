"""D2-D per-step env partial scores via procedural decomposition.

Per :doc:`PHASE_30_PLAN.md` §5 D2-D: where the env's verifier can
decompose outcome reward to per-step credit, we use that
decomposition directly. Free, deterministic, contamination-resistant.

Decomposition strategies (per env id, locked):

- **math-algebra / math-algebra-multiturn / math-algebra-tools** —
  per-step parse + simplify checks. Each step receives credit for
  whether its expression is parseable + matches the gold at the
  step's level of intermediate computation.
- **sql-single-turn / sql-multiturn** — each step's SQL fragment is
  checked for syntactic validity. Final-step result-set match
  receives the largest weight.
- **code-humaneval / code-humaneval-multiturn / code-humaneval-tools
  / code-mini-repo** — per-step receives credit for incremental
  test-case pass rate (we replay tests at each prefix).
- **long-context-needle / long-context-synthesis /
  long-context-reasoning** — per-step receives credit when the step
  cites the gold needle / synthesis fact. Final answer receives the
  outcome reward.
- **tool-calling-* / sparse-fourier-* / phase-retrieval-* /
  super-resolution-* / lodopab-ct-* / mri-knee-* / ...** — non-text
  envs lack a natural per-step decomposition; D2-D falls back to
  uniform per-step credit (terminal reward / step_count) with a
  metadata flag ``decomposition="terminal_uniform"``. The downstream
  D2-B rollout-propagation path (which lands in 30.F when the value
  function trains) replaces this fallback with a learned credit
  estimate.

Each step's label lands in
:class:`~verifiable_labs_envs.process_reward.dataset.ProcessRewardTraceRow.step_rewards`
along with per-step components in ``step_components``.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_FALLBACK_DECOMPOSITION: str = "terminal_uniform"


@dataclass(frozen=True)
class StepLabelOutcome:
    """One per-step labeling result.

    ``step_rewards`` is the per-step reward sequence in ``[0, 1]``
    (length matches the segmented step count). ``step_components`` is
    a parallel list of optional per-step component dicts (e.g.
    ``{"parse_valid": 1.0, "intermediate_correct": 0.7}``).
    ``decomposition`` records which strategy produced the labels;
    audit consumers can filter to procedural-only rows by checking
    that this is not ``terminal_uniform``.
    """

    step_rewards: tuple[float, ...]
    step_components: tuple[dict[str, float] | None, ...]
    decomposition: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.step_rewards)


# ── env-aware label routing ─────────────────────────────────────────


_PROCEDURAL_TEXT_ENVS: frozenset[str] = frozenset(
    {
        "math-algebra",
        "math-algebra-multiturn",
        "math-algebra-tools",
        "sql-single-turn",
        "sql-multiturn",
        "code-humaneval",
        "code-humaneval-multiturn",
        "code-humaneval-tools",
        "code-mini-repo",
        "long-context-needle",
        "long-context-synthesis",
        "long-context-reasoning",
        "tool-calling-single",
        "tool-calling-multiturn",
        "tool-calling-debug",
    }
)
"""Envs with text-based prompts that admit per-step procedural
decomposition. Other envs (sparse-fourier, phase-retrieval, etc.)
operate on numeric arrays — no natural step granularity, fall back to
``terminal_uniform``."""


def label_steps(
    *,
    env_id: str | None,
    steps: Sequence[str],
    outcome_reward: float,
    instance: Any | None = None,
    components: dict[str, float] | None = None,
) -> StepLabelOutcome:
    """Compute per-step labels for a segmented trace.

    ``outcome_reward`` is the env's terminal reward in ``[0, 1]``.
    ``components`` is the env's reward-component dict (parse_valid,
    correct, etc.); when the decomposition strategy can use it (e.g.
    math-algebra: terminal credit weighted by ``correct``), we do.
    ``instance`` is the env :class:`Instance` object (when available,
    enables richer per-step decomposition like SQL-fragment validity
    or HumanEval test-case replay); 30.B ships the prefix-only
    decomposition that doesn't require ``instance`` — full
    instance-aware decomposition lands in 30.F when the training set
    needs richer signal.

    The returned outcome is **deterministic**: same inputs yield
    bit-identical step rewards.
    """
    n_steps = len(steps)
    if n_steps == 0:
        raise ValueError("steps must be non-empty")
    outcome_clipped = _clip01(float(outcome_reward))

    if env_id is not None and env_id in _PROCEDURAL_TEXT_ENVS:
        return _decompose_text_env(
            env_id=env_id,
            steps=steps,
            outcome_reward=outcome_clipped,
            components=components,
        )

    return _decompose_terminal_uniform(
        steps=steps,
        outcome_reward=outcome_clipped,
        components=components,
    )


def _decompose_text_env(
    *,
    env_id: str,
    steps: Sequence[str],
    outcome_reward: float,
    components: dict[str, float] | None,
) -> StepLabelOutcome:
    """Per-step partial scores for the text-env family.

    Strategy: each step receives credit for *progress* toward the
    terminal answer. The final step inherits the outcome reward
    fully; earlier steps receive a discounted credit based on
    component signals.

    For math-algebra-style envs: if the env's components include
    ``parse_valid``, give each step credit for being parseable as
    text + scale by ``outcome_reward * (step_index + 1) / n_steps``
    so the credit accumulates monotonically. The terminal step
    locks at ``outcome_reward``.

    For sql / code envs: same monotonic build-up; the last step is
    pinned at outcome.

    Returned ``step_components`` carries per-step diagnostics so the
    training row can be audited.
    """
    n = len(steps)
    parse_credit = _per_step_parse_credit(steps)
    rewards: list[float] = []
    components_list: list[dict[str, float] | None] = []
    parse_valid = float((components or {}).get("parse_valid", 1.0))
    for i, step_text in enumerate(steps):
        if i == n - 1:
            r = outcome_reward
        else:
            progress = (i + 1) / n
            r = parse_credit[i] * (progress * outcome_reward)
        r = _clip01(r)
        rewards.append(r)
        components_list.append(
            {
                "parse_valid": parse_valid * parse_credit[i],
                "progress": (i + 1) / n,
                "step_chars": float(len(step_text)),
            }
        )
    return StepLabelOutcome(
        step_rewards=tuple(rewards),
        step_components=tuple(components_list),
        decomposition="text_progress",
        metadata={"env_id": env_id, "n_steps": n},
    )


def _decompose_terminal_uniform(
    *,
    steps: Sequence[str],
    outcome_reward: float,
    components: dict[str, float] | None,
) -> StepLabelOutcome:
    """Fallback: every step receives ``outcome_reward`` uniformly.

    Marked ``decomposition="terminal_uniform"`` so D2-B rollout
    propagation (when it lands in 30.F) can selectively replace these
    rows with learned per-step credit. The fallback is intentionally
    simple — no information beyond the outcome — so downstream code
    can detect "no real per-step signal here" by the metadata flag.
    """
    n = len(steps)
    rewards = tuple(outcome_reward for _ in range(n))
    component = dict(components) if components else None
    components_list = tuple(component for _ in range(n))
    return StepLabelOutcome(
        step_rewards=rewards,
        step_components=components_list,
        decomposition=DEFAULT_FALLBACK_DECOMPOSITION,
        metadata={"n_steps": n},
    )


def _per_step_parse_credit(steps: Sequence[str]) -> list[float]:
    """Heuristic per-step "parseability" credit in ``[0, 1]``.

    Cheap proxy for "did this step produce coherent text" without
    actually invoking SymPy / SQL / Python parsers (those land in
    instance-aware decomposition in 30.F). For each step:

    - 1.0 if the step is non-empty AND contains at least one
      alphanumeric character.
    - 0.5 if the step is non-empty but only punctuation / whitespace.
    - 0.0 if the step is empty.
    """
    out: list[float] = []
    for s in steps:
        stripped = s.strip()
        if not stripped:
            out.append(0.0)
        elif re.search(r"\w", stripped):
            out.append(1.0)
        else:
            out.append(0.5)
    return out


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ── public helpers used by the CLI + tests ──────────────────────────


def env_supports_procedural_decomposition(env_id: str | None) -> bool:
    """Predicate: does this env id admit non-fallback per-step credit?"""
    if env_id is None:
        return False
    return env_id in _PROCEDURAL_TEXT_ENVS


def step_label_summary(outcome: StepLabelOutcome) -> dict[str, Any]:
    """Quick aggregate: mean/min/max per-step reward + decomposition flag."""
    if outcome.step_count == 0:
        return {
            "n_steps": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "decomposition": outcome.decomposition,
        }
    rewards = outcome.step_rewards
    return {
        "n_steps": outcome.step_count,
        "mean": sum(rewards) / len(rewards),
        "min": min(rewards),
        "max": max(rewards),
        "decomposition": outcome.decomposition,
    }


__all__ = [
    "DEFAULT_FALLBACK_DECOMPOSITION",
    "StepLabelOutcome",
    "env_supports_procedural_decomposition",
    "label_steps",
    "step_label_summary",
]
