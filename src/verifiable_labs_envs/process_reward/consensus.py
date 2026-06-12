"""Per-step + aggregate consensus reward (Phase 30.B, plan §5 D5-D
extended to step granularity).

Phase 29's :func:`verifiable_labs_envs.reward_distillation.consensus.consensus_reward`
blends the env-procedural reward (D5-A) with the optional frontier
judgment (D5-C) at the **trace level**. Phase 30 extends the same
70/30 blend to **per-step** granularity:

```python
per_step_consensus(env_step_rewards, frontier_step_rewards)
# applies the 70/30 D5-D blend per step;
# falls back to env-only when the frontier signal is None for that step.
```

All weights default to the locked
:data:`~verifiable_labs_envs.reward_distillation.consensus.DEFAULT_ENV_WEIGHT`
/
:data:`~verifiable_labs_envs.reward_distillation.consensus.DEFAULT_FRONTIER_WEIGHT`
(0.7 / 0.3) so the moat-aligned env signal stays dominant.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from verifiable_labs_envs.reward_distillation.consensus import (
    DEFAULT_ENV_WEIGHT,
    DEFAULT_FRONTIER_WEIGHT,
    consensus_reward,
)


def per_step_consensus(
    env_step_rewards: Sequence[float | None],
    frontier_step_rewards: Sequence[float | None] | None = None,
    *,
    env_weight: float = DEFAULT_ENV_WEIGHT,
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT,
) -> tuple[float, ...]:
    """Per-step blend.

    Each output entry is computed via the Phase 29
    :func:`consensus_reward` helper applied to the matching pair from
    ``env_step_rewards`` / ``frontier_step_rewards``. If
    ``frontier_step_rewards`` is ``None`` (no judge slice) every step
    falls back to its env reward.

    ``env_step_rewards`` and ``frontier_step_rewards`` (when provided)
    must have the same length. ``None`` entries on the env side mean
    "no signal" and propagate from the frontier side; if both sides
    are ``None`` for a step, the step receives ``0.5`` (neutral
    sentinel — auditable and clipped to-the-unit interval).
    """
    n = len(env_step_rewards)
    if frontier_step_rewards is not None and len(frontier_step_rewards) != n:
        raise ValueError(
            f"length mismatch: env_step_rewards={n}, "
            f"frontier_step_rewards={len(frontier_step_rewards)}"
        )

    out: list[float] = []
    for i in range(n):
        env_r = env_step_rewards[i]
        front_r = (
            frontier_step_rewards[i]
            if frontier_step_rewards is not None
            else None
        )
        if env_r is None and front_r is None:
            out.append(0.5)
            continue
        out.append(
            consensus_reward(
                env_r,
                front_r,
                env_weight=env_weight,
                frontier_weight=frontier_weight,
            )
        )
    return tuple(out)


def trace_aggregate_consensus(
    step_consensus: Sequence[float],
    *,
    env_outcome: float | None = None,
    method: str = "mean",
    env_weight: float = DEFAULT_ENV_WEIGHT,
    frontier_weight: float = DEFAULT_FRONTIER_WEIGHT,
) -> float:
    """Aggregate score across the per-step consensus rewards.

    Two methods supported:

    - ``"mean"`` (default) — arithmetic mean of step consensus.
    - ``"env_blend"`` — when an explicit ``env_outcome`` is supplied,
      blend the per-step mean (treated as the "frontier" signal here)
      with the env's terminal outcome (treated as the "env" signal):
      this preserves the 70/30 trace-level blend on top of the
      per-step blend.

    Returns a scalar in ``[0, 1]``.
    """
    if not step_consensus:
        return 0.5

    n = len(step_consensus)
    mean = sum(float(r) for r in step_consensus) / n

    if method == "mean":
        return _clip01(mean)

    if method == "env_blend":
        if env_outcome is None:
            return _clip01(mean)
        return consensus_reward(
            env_outcome,
            mean,
            env_weight=env_weight,
            frontier_weight=frontier_weight,
        )

    raise ValueError(f"unknown aggregation method: {method!r}")


def per_step_disagreement(
    env_step_rewards: Sequence[float | None],
    frontier_step_rewards: Sequence[float | None],
) -> tuple[float | None, ...]:
    """Per-step ``|env − frontier|``; ``None`` where either side is missing.

    Used by 30.B's downstream code to flag borderline steps for the
    frontier-judge slice (D2-C).
    """
    if len(env_step_rewards) != len(frontier_step_rewards):
        raise ValueError(
            f"length mismatch: env_step_rewards={len(env_step_rewards)}, "
            f"frontier_step_rewards={len(frontier_step_rewards)}"
        )
    out: list[float | None] = []
    for env_r, front_r in zip(env_step_rewards, frontier_step_rewards, strict=True):
        if env_r is None or front_r is None:
            out.append(None)
        else:
            out.append(abs(float(env_r) - float(front_r)))
    return tuple(out)


def per_step_disagreement_metrics(rows: Iterable[Any]) -> dict[str, float]:
    """Aggregate per-step disagreement across a row collection.

    Each row must expose a ``step_disagreements`` attribute or
    ``["step_disagreements"]`` key holding a tuple/list of floats or
    ``None`` entries. Missing rows / ``None`` per-step entries are
    skipped.

    Returns a dict with:

    - ``count``: number of measurable per-step disagreement values.
    - ``trace_count``: number of rows contributing at least one value.
    - ``mean``, ``max``, ``min``, ``p50``, ``p90``: scalar summaries.
    """
    values: list[float] = []
    contributing_traces = 0
    for row in rows:
        seq = _maybe_step_disagreements(row)
        if seq is None:
            continue
        before = len(values)
        for v in seq:
            if v is None:
                continue
            values.append(float(v))
        if len(values) > before:
            contributing_traces += 1

    if not values:
        return {
            "count": 0.0,
            "trace_count": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)
    return {
        "count": float(n),
        "trace_count": float(contributing_traces),
        "mean": sum(sorted_values) / n,
        "max": sorted_values[-1],
        "min": sorted_values[0],
        "p50": _linear_quantile(sorted_values, 0.50),
        "p90": _linear_quantile(sorted_values, 0.90),
    }


def borderline_step_indices(
    env_step_rewards: Sequence[float | None],
    *,
    low: float = 0.3,
    high: float = 0.7,
) -> list[int]:
    """Indices of *steps* whose env reward lies in the borderline
    window ``(low, high)``.

    Used by :mod:`verifiable_labs_envs.process_reward.frontier_judge`
    to pick which steps to send to the frontier judge — only middle-band
    steps benefit from a second opinion (the tails are already
    informative).
    """
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f"require 0 <= low < high <= 1; got low={low}, high={high}"
        )
    return [
        i
        for i, r in enumerate(env_step_rewards)
        if r is not None and low < float(r) < high
    ]


# ── internals ───────────────────────────────────────────────────────


def _maybe_step_disagreements(row: Any) -> Sequence | None:
    if hasattr(row, "step_disagreements"):
        seq = row.step_disagreements
    elif isinstance(row, dict) and "step_disagreements" in row:
        seq = row["step_disagreements"]
    else:
        return None
    if seq is None:
        return None
    if not hasattr(seq, "__iter__"):
        return None
    return seq


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _linear_quantile(sorted_values: list[float], q: float) -> float:
    """Empirical ``q``-quantile via linear interpolation."""
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
    "borderline_step_indices",
    "per_step_consensus",
    "per_step_disagreement",
    "per_step_disagreement_metrics",
    "trace_aggregate_consensus",
]
