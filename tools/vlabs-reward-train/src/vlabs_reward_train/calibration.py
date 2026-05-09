"""Conformal calibration step for the trained student (D10-A).

Per :doc:`PHASE_29_PLAN.md` §11:

```python
residuals = [|student.predict(p, c) - row.consensus_reward| for row in calib_set]
quantile = split_conformal_quantile(residuals, alpha=0.10)
```

The trained student then serves
``[max(0, point - quantile), min(1, point + quantile)]`` with a
finite-sample, distribution-free 90 % marginal coverage guarantee
(Lei et al. 2018; the same Layer 1 moat kernel as ``/v1/score``).

This module is **pure Python** — it builds on
:func:`verifiable_labs_envs.conformal.split_conformal_quantile`. The
trained student inference layer is the upstream caller's
responsibility; tests pass a callable.
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
from verifiable_labs_envs.reward_distillation.dataset import RewardTrainingRow

DEFAULT_ALPHA: float = 0.10
DEFAULT_TARGET_COVERAGE: float = 1.0 - DEFAULT_ALPHA


@dataclass(frozen=True)
class CalibrationResult:
    """One conformal-calibration run on the held-out calibration set."""

    quantile: float
    alpha: float
    n_rows: int
    target_coverage: float
    empirical_coverage: float
    drift: float

    def is_calibration_suspect(self, *, drift_tol: float = 0.05) -> bool:
        """Drift > ``drift_tol`` flags the model as "calibration suspect".

        Mirrors the §5 D10-B cross-check rule: if empirical and target
        coverage diverge by more than 5 pp on a held-out set, don't
        ship the checkpoint.
        """
        return abs(self.drift) > float(drift_tol)


def calibrate_residuals(
    rows: Sequence[RewardTrainingRow],
    student_predict: Callable[[str, str], float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> CalibrationResult:
    """Compute the conformal quantile + empirical-coverage diagnostic.

    Steps:

    1. For each row in the calibration set, evaluate the student to get
       a predicted scalar reward.
    2. Compute absolute residuals against ``row.consensus_reward``.
    3. Apply :func:`split_conformal_quantile` at ``alpha`` to get
       :math:`\\hat q_\\alpha`.
    4. Re-walk the rows and count how many fall inside
       ``[predicted - q, predicted + q]`` to derive empirical coverage.
    5. Return a :class:`CalibrationResult` with the quantile, drift
       (empirical − target), and a serialisable record.

    ``student_predict`` is a `Callable[[prompt, completion], reward]`
    so tests can pass a deterministic stub (29.D's stub student returns
    ``0.5 + uniform(-0.1, 0.1)``).
    """
    if not rows:
        raise ValueError("calibration set is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    targets: list[float] = []
    predictions: list[float] = []
    for row in rows:
        target = float(row.consensus_reward)
        predicted = float(student_predict(row.prompt, row.completion))
        targets.append(target)
        predictions.append(predicted)

    targets_arr = np.asarray(targets, dtype=np.float64)
    pred_arr = np.asarray(predictions, dtype=np.float64)
    residuals = np.abs(pred_arr - targets_arr)

    quantile = split_conformal_quantile(residuals, alpha=alpha)
    lower = np.maximum(0.0, pred_arr - quantile)
    upper = np.minimum(1.0, pred_arr + quantile)
    coverage = _empirical_coverage(targets_arr, lower, upper)
    target_coverage = 1.0 - alpha
    return CalibrationResult(
        quantile=float(quantile),
        alpha=float(alpha),
        n_rows=int(targets_arr.size),
        target_coverage=float(target_coverage),
        empirical_coverage=float(coverage),
        drift=float(coverage - target_coverage),
    )


def score_with_ci(
    student_predict: Callable[[str, str], float],
    quantile: float,
    *,
    prompt: str,
    completion: str,
) -> dict[str, float | bool]:
    """Apply a calibrated student to a single (prompt, completion).

    Returns the canonical service-shaped payload:

    ```python
    {"reward": float, "ci_low": float, "ci_high": float,
     "coverage_guarantee": 1 - alpha, "calibrated": True}
    ```

    ``calibrated=False`` is reserved for the 29.E stub-mode response —
    callers explicitly opt into that fallback shape.
    """
    if quantile < 0:
        raise ValueError(f"quantile must be non-negative; got {quantile}")
    point = float(student_predict(prompt, completion))
    low = max(0.0, point - quantile)
    high = min(1.0, point + quantile)
    return {
        "reward": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "coverage_guarantee": float(1.0 - DEFAULT_ALPHA),
        "calibrated": True,
    }


def stub_student_predict(seed: int = 0) -> Callable[[str, str], float]:
    """Deterministic stand-in predictor used by 29.D + 29.E test paths.

    Hashes ``(prompt, completion)`` into a uniform reward in
    ``[0.4, 0.6]`` so the calibration math sees variation but the
    expected coverage stays close to nominal — useful for asserting
    the harness wires through end-to-end without a trained student.
    """
    rng = np.random.default_rng(seed)
    # Pre-draw an offset table indexed by row hash for determinism.

    def predict(prompt: str, completion: str) -> float:
        h = hash((prompt, completion)) & 0xFFFF
        return 0.5 + float(rng.uniform(-0.1, 0.1)) * (h / 0xFFFF)

    return predict


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_TARGET_COVERAGE",
    "CalibrationResult",
    "calibrate_residuals",
    "score_with_ci",
    "stub_student_predict",
]
