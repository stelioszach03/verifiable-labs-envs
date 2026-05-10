"""D9-C per-step + aggregate conformal calibration (Phase 30.D).

Per :doc:`PHASE_30_PLAN.md` §11:

```python
# For each step position bucket b = (0..1), (1..3), (3..7), (7..32):
#   residuals_b = [|student.predict_step(prompt, steps, t) - true_step_reward[t]|
#                  for row in calib_set
#                  for t in bucket
#                  if t < row.step_count and row.step_rewards[t] is not None]
#   q_b = split_conformal_quantile(residuals_b, alpha=0.10)

# Aggregate quantile over trace-level residuals:
# residuals_agg = [|student.predict_aggregate(prompt, steps) - true_aggregate|
#                  for row in calib_set]
# q_agg = split_conformal_quantile(residuals_agg, alpha=0.10)
```

This module is **pure Python** — it builds on
:func:`verifiable_labs_envs.conformal.split_conformal_quantile` and
the existing Phase 29 conformal kernel.

The trained student inference layer is the upstream caller's
responsibility; tests pass deterministic step + aggregate predictors
from
:mod:`verifiable_labs_envs.process_reward.inference`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from verifiable_labs_envs.conformal import (
    coverage as _empirical_coverage,
)
from verifiable_labs_envs.conformal import (
    split_conformal_quantile,
)
from verifiable_labs_envs.process_reward.dataset import (
    ProcessRewardTraceRow,
)

DEFAULT_ALPHA: float = 0.10
DEFAULT_TARGET_COVERAGE: float = 1.0 - DEFAULT_ALPHA
DEFAULT_DRIFT_TOL: float = 0.05
"""Plan §5 D9-C: aggregate empirical coverage must land within ±5 pp
of target."""


DEFAULT_POSITION_BUCKETS: tuple[range, ...] = (
    range(0, 1),
    range(1, 3),
    range(3, 7),
    range(7, 32),
)
"""Per :doc:`PHASE_30_PLAN.md` §11 — coarse step-position bucketing
for per-step calibration. Mitigates R8 (per-call O(1) lookup at
serve time)."""


# ── result dataclasses ─────────────────────────────────────────────


@dataclass(frozen=True)
class PerStepBucketResult:
    """One per-step-bucket calibration result."""

    bucket_label: str
    n_residuals: int
    quantile: float


@dataclass(frozen=True)
class CalibrationResult:
    """Combined per-step + aggregate calibration result."""

    per_step_quantiles: dict[str, float]
    per_step_bucket_results: tuple[PerStepBucketResult, ...]
    aggregate_quantile: float
    aggregate_target_coverage: float
    aggregate_empirical_coverage: float
    aggregate_drift: float
    n_traces: int
    alpha: float

    def is_calibration_suspect(self, *, drift_tol: float = DEFAULT_DRIFT_TOL) -> bool:
        """Predicate: does the aggregate empirical coverage drift
        past ``drift_tol``?

        Mirrors :doc:`PHASE_30_PLAN.md` D10-B cross-check rule
        (inherited from Phase 29).
        """
        return abs(self.aggregate_drift) > float(drift_tol)


@dataclass(frozen=True)
class PerStepCoverageReport:
    """Per-step-bucket coverage diagnostic.

    Returned by :func:`evaluate_per_step_coverage`. Aggregate-level
    coverage is the gate; per-position drift is logged but not hard-
    gating (rare positions have wide noise).
    """

    by_bucket: dict[str, float]
    overall: float
    n_residuals_total: int


# ── core calibration ───────────────────────────────────────────────


def calibrate_residuals(
    rows: Sequence[ProcessRewardTraceRow],
    step_predictor: Callable[[str, Sequence[str], int], float],
    aggregate_predictor: Callable[[str, Sequence[str]], float],
    *,
    alpha: float = DEFAULT_ALPHA,
    position_buckets: Sequence[range] = DEFAULT_POSITION_BUCKETS,
) -> CalibrationResult:
    """Fit per-position-bucket conformal quantiles + an aggregate
    quantile against a held-out calibration set.

    Inputs:

    - ``rows``: held-out calibration trace rows.
    - ``step_predictor``: ``(prompt, steps, step_index) -> reward``.
      The trained student arrives in 30.G; tests use the deterministic
      stub from
      :mod:`verifiable_labs_envs.process_reward.inference`.
    - ``aggregate_predictor``: ``(prompt, steps) -> reward``.
    - ``alpha``: target miscoverage fraction (default 0.10 ⇒ 90%
      coverage).
    - ``position_buckets``: locked 4 buckets per §11.

    Returns the full :class:`CalibrationResult` with per-step
    quantiles dict, per-bucket records, aggregate quantile, and
    drift diagnostic.
    """
    if not rows:
        raise ValueError("calibration set is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if not position_buckets:
        raise ValueError("position_buckets must be non-empty")

    bucket_records: list[PerStepBucketResult] = []
    bucket_quantiles: dict[str, float] = {}
    for bucket in position_buckets:
        residuals: list[float] = []
        for row in rows:
            for t in bucket:
                if t >= row.step_count:
                    continue
                env_r = row.step_rewards[t]
                # Fall back to consensus_step_reward when env is None.
                target = (
                    float(row.step_consensus_rewards[t])
                    if env_r is None
                    else float(env_r)
                )
                pred = float(step_predictor(row.prompt, row.steps, t))
                residuals.append(abs(pred - target))
        if not residuals:
            continue
        q = float(split_conformal_quantile(np.asarray(residuals), alpha=alpha))
        label = bucket_label(bucket)
        bucket_records.append(
            PerStepBucketResult(
                bucket_label=label,
                n_residuals=len(residuals),
                quantile=q,
            )
        )
        bucket_quantiles[label] = q

    # Aggregate calibration.
    agg_targets = np.asarray(
        [float(r.aggregate_reward) for r in rows], dtype=np.float64
    )
    agg_predictions = np.asarray(
        [float(aggregate_predictor(r.prompt, r.steps)) for r in rows],
        dtype=np.float64,
    )
    agg_residuals = np.abs(agg_predictions - agg_targets)
    agg_quantile = float(split_conformal_quantile(agg_residuals, alpha=alpha))
    lower = np.maximum(0.0, agg_predictions - agg_quantile)
    upper = np.minimum(1.0, agg_predictions + agg_quantile)
    coverage = _empirical_coverage(agg_targets, lower, upper)
    target_coverage = 1.0 - alpha

    return CalibrationResult(
        per_step_quantiles=bucket_quantiles,
        per_step_bucket_results=tuple(bucket_records),
        aggregate_quantile=agg_quantile,
        aggregate_target_coverage=target_coverage,
        aggregate_empirical_coverage=float(coverage),
        aggregate_drift=float(coverage) - target_coverage,
        n_traces=len(rows),
        alpha=float(alpha),
    )


def evaluate_per_step_coverage(
    rows: Sequence[ProcessRewardTraceRow],
    step_predictor: Callable[[str, Sequence[str], int], float],
    *,
    per_step_quantiles: dict[str, float],
    position_buckets: Sequence[range] = DEFAULT_POSITION_BUCKETS,
) -> PerStepCoverageReport:
    """Empirical per-step CI coverage diagnostic.

    For each step position, look up the bucket's quantile and check
    whether the per-step target falls inside
    ``[predicted - q, predicted + q]`` clipped to ``[0, 1]``.
    """
    by_bucket: dict[str, float] = {}
    total_hits = 0
    total_residuals = 0
    for bucket in position_buckets:
        label = bucket_label(bucket)
        q = per_step_quantiles.get(label)
        if q is None:
            continue
        hits = 0
        n = 0
        for row in rows:
            for t in bucket:
                if t >= row.step_count:
                    continue
                env_r = row.step_rewards[t]
                target = (
                    float(row.step_consensus_rewards[t])
                    if env_r is None
                    else float(env_r)
                )
                pred = float(step_predictor(row.prompt, row.steps, t))
                low = max(0.0, pred - q)
                high = min(1.0, pred + q)
                if low <= target <= high:
                    hits += 1
                n += 1
        if n > 0:
            by_bucket[label] = hits / n
            total_hits += hits
            total_residuals += n
    overall = (total_hits / total_residuals) if total_residuals > 0 else 0.0
    return PerStepCoverageReport(
        by_bucket=by_bucket,
        overall=overall,
        n_residuals_total=total_residuals,
    )


# ── helpers ────────────────────────────────────────────────────────


def bucket_label(bucket: range) -> str:
    """Stable string label for a bucket (used as JSONB key in
    `process_reward_models.step_conformal_quantiles`)."""
    return f"range({bucket.start}, {bucket.stop})"


def score_with_ci(
    *,
    prompt: str,
    steps: Sequence[str],
    step_predictor: Callable[[str, Sequence[str], int], float],
    aggregate_predictor: Callable[[str, Sequence[str]], float],
    per_step_quantiles: dict[str, float],
    aggregate_quantile: float,
    position_buckets: Sequence[range] = DEFAULT_POSITION_BUCKETS,
) -> dict[str, list | float | bool]:
    """Apply calibrated quantiles at serve-time to produce the
    canonical PRM service-shaped payload.

    Returns the per-step + aggregate score envelope ready to wire into
    the 30.E response shape. ``calibrated=True`` flags that this is
    the real path (vs the stub in 30.E which returns
    ``calibrated=False`` until 30.G).
    """
    if aggregate_quantile < 0:
        raise ValueError(
            f"aggregate_quantile must be non-negative; got {aggregate_quantile}"
        )
    if not steps:
        raise ValueError("steps must be non-empty")

    step_rewards: list[float] = []
    step_cis: list[list[float]] = []
    for i, _step in enumerate(steps):
        bucket_q = _bucket_quantile_for_position(
            i, position_buckets=position_buckets, per_step_quantiles=per_step_quantiles
        )
        pred = float(step_predictor(prompt, steps, i))
        low = max(0.0, pred - bucket_q)
        high = min(1.0, pred + bucket_q)
        step_rewards.append(pred)
        step_cis.append([low, high])

    agg_pred = float(aggregate_predictor(prompt, steps))
    agg_low = max(0.0, agg_pred - aggregate_quantile)
    agg_high = min(1.0, agg_pred + aggregate_quantile)
    return {
        "step_rewards": step_rewards,
        "step_confidence_intervals": step_cis,
        "aggregate_reward": agg_pred,
        "aggregate_confidence_interval": [agg_low, agg_high],
        "coverage_guarantee": float(1.0 - DEFAULT_ALPHA),
        "calibrated": True,
    }


def _bucket_quantile_for_position(
    position: int,
    *,
    position_buckets: Sequence[range],
    per_step_quantiles: dict[str, float],
) -> float:
    """Look up the bucket containing ``position`` and return its
    quantile. Falls back to the largest available bucket's quantile
    when the position lies past every defined bucket — the safe
    conservative choice."""
    for bucket in position_buckets:
        if position in bucket:
            label = bucket_label(bucket)
            if label in per_step_quantiles:
                return per_step_quantiles[label]
    if per_step_quantiles:
        # Fallback to the largest quantile across the dict — safer
        # over-coverage than under-coverage on unseen positions.
        return max(per_step_quantiles.values())
    return 0.0


def position_to_bucket_label(
    position: int,
    *,
    position_buckets: Sequence[range] = DEFAULT_POSITION_BUCKETS,
) -> str | None:
    """Public helper exposed for the 30.E API surface — given a step
    position, return the bucket label that owns it (or ``None`` if no
    bucket covers the position)."""
    for bucket in position_buckets:
        if position in bucket:
            return bucket_label(bucket)
    return None


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DRIFT_TOL",
    "DEFAULT_POSITION_BUCKETS",
    "DEFAULT_TARGET_COVERAGE",
    "CalibrationResult",
    "PerStepBucketResult",
    "PerStepCoverageReport",
    "bucket_label",
    "calibrate_residuals",
    "evaluate_per_step_coverage",
    "position_to_bucket_label",
    "score_with_ci",
]
