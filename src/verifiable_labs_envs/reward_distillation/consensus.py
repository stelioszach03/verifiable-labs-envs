"""D5-D consensus reward and disagreement metrics (Phase 29.B).

Per :doc:`PHASE_29_PLAN.md` §5 D5-D, every distilled-reward training
row is built by blending the *primary* env-procedural reward with an
*optional* frontier-model judgment so the student model is anchored
on the Layer 1 conformal moat while still learning instruction-following
texture from the small frontier slice.

The blend favours the env signal 70/30 because the env's procedural
verifier is the calibration-trustworthy source; the frontier judge
is a breadth supplement, not a replacement.

Disagreement is intentionally *retained* as information: rows where
``|env_reward - frontier_judgment|`` is large feed the calibration set
in D10 to widen the conformal interval where the underlying signal is
noisy. A single-signal student would have to *fake* a confidence
interval; ours is empirically calibrated against measured label noise.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_ENV_WEIGHT: float = 0.7
DEFAULT_FRONTIER_WEIGHT: float = 0.3


def consensus_reward(
    env_reward: float | None,
    frontier_reward: float | None,
    *,
    env_weight: float = DEFAULT_ENV_WEIGHT,
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT,
) -> float:
    """Blend env-procedural reward and frontier-model judgment into the
    scalar training target.

    Behaviour:

    - Both signals present → ``env_weight * env + frontier_weight * frontier``.
    - Only ``env_reward`` present (the common case, ~90 % of rows) →
      return ``env_reward`` unchanged (the Layer 1 moat takes precedence).
    - Only ``frontier_reward`` present (external rows from UltraFeedback
      where there is no env score to anchor on) → return ``frontier_reward``.
    - Both ``None`` → :class:`ValueError`. Rows must carry at least one
      reward source; the dataset builder is responsible for filtering.

    All inputs are clipped to ``[0, 1]`` after blending — both env scores
    and frontier judgments are nominally in that range, but defensive
    clipping protects against malformed external rows leaking in.

    Weights default to the locked D5-D 70/30 ratio. Callers may override
    for ablation sweeps in Phase 29.F-G; both weights must be non-negative
    and their sum must be > 0.
    """
    if env_weight < 0 or frontier_weight < 0:
        raise ValueError(
            f"weights must be non-negative; got env={env_weight}, "
            f"frontier={frontier_weight}"
        )
    total_weight = env_weight + frontier_weight
    if total_weight <= 0:
        raise ValueError("env_weight + frontier_weight must be > 0")

    if env_reward is None and frontier_reward is None:
        raise ValueError("at least one of env_reward or frontier_reward must be set")

    if frontier_reward is None:
        return _clip01(float(env_reward))  # type: ignore[arg-type]
    if env_reward is None:
        return _clip01(float(frontier_reward))

    blended = (env_weight * float(env_reward) + frontier_weight * float(frontier_reward)) / total_weight
    return _clip01(blended)


def disagreement(env_reward: float, frontier_reward: float) -> float:
    """Absolute disagreement ``|env_reward - frontier_reward|``.

    Both inputs assumed in ``[0, 1]``; the result is in ``[0, 1]`` and
    feeds the per-row ``disagreement`` column of the dataset.
    """
    return abs(float(env_reward) - float(frontier_reward))


def disagreement_metrics(rows: Iterable[Any]) -> dict[str, float]:
    """Aggregate disagreement statistics across a row collection.

    Each row must expose a ``disagreement`` attribute or ``["disagreement"]``
    key holding either a float or ``None``. ``None`` and missing rows are
    skipped — those carry no frontier judgment, so disagreement is
    undefined.

    Returns a dict with:

    - ``count``: number of rows with measurable disagreement.
    - ``mean``, ``max``, ``min``: scalar summaries.
    - ``p25``, ``p50``, ``p75``: empirical quartiles via linear
      interpolation (``numpy.quantile`` would be the obvious choice but
      we keep this module pure-Python so the import graph stays light).
    """
    values: list[float] = []
    for row in rows:
        value = _maybe_disagreement(row)
        if value is not None:
            values.append(float(value))

    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)
    return {
        "count": float(n),
        "mean": sum(sorted_values) / n,
        "max": sorted_values[-1],
        "min": sorted_values[0],
        "p25": _linear_quantile(sorted_values, 0.25),
        "p50": _linear_quantile(sorted_values, 0.50),
        "p75": _linear_quantile(sorted_values, 0.75),
    }


def borderline_indices(
    env_rewards: Sequence[float],
    *,
    low: float = 0.3,
    high: float = 0.7,
) -> list[int]:
    """Indices of rows whose env reward lies in the borderline window
    ``(low, high)``.

    Used by :mod:`verifiable_labs_envs.reward_distillation.frontier_judge`
    to pick which rows to send to the frontier judge: the easy wins
    (env_reward near 0 or 1) don't need a second opinion, while the
    middle-band rows are where labelling noise is highest and where
    blended consensus gains the most.
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"require 0 <= low < high <= 1; got low={low}, high={high}")
    return [i for i, r in enumerate(env_rewards) if low < float(r) < high]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _maybe_disagreement(row: Any) -> float | None:
    if hasattr(row, "disagreement"):
        return row.disagreement
    if isinstance(row, dict) and "disagreement" in row:
        return row["disagreement"]
    return None


def _linear_quantile(sorted_values: list[float], q: float) -> float:
    """Empirical ``q``-quantile via linear interpolation between order
    statistics. Mirrors ``numpy.quantile(..., method="linear")`` for the
    pure-Python path."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    frac = pos - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


__all__ = [
    "DEFAULT_ENV_WEIGHT",
    "DEFAULT_FRONTIER_WEIGHT",
    "borderline_indices",
    "consensus_reward",
    "disagreement",
    "disagreement_metrics",
]
