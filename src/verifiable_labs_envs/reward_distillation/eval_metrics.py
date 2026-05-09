"""Eval metrics for the distilled reward model (Phase 29.D, plan §9).

Three metric families:

- **Spearman ρ** — primary signal-quality metric on the held-out env
  test sets (D7-A pass criterion: ρ ≥ 0.70).
- **MAE / bias** — standard regression accuracy summaries.
- **Coverage / calibration drift** — D10 moat metric, the
  finite-sample coverage of the student's CI vs the target ``1 - α``.

All implementations are pure-numpy / pure-Python so the harness stays
import-light + GPU-free in 29.D.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankCorrelation:
    """Spearman ρ + p-value approximation result."""

    rho: float
    n: int

    @property
    def is_significant(self) -> bool:
        """Loose proxy: ρ × √(n - 1) > 1.96 (one-sided 95 %)."""
        if self.n < 3:
            return False
        return abs(self.rho) * (self.n - 1) ** 0.5 > 1.96


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> RankCorrelation:
    """Pure-numpy Spearman rank correlation (no scipy dependency).

    Returns ρ = 0 for inputs shorter than 2 or constant-ranked
    inputs (where σ(rank) = 0). Used by both 29.D's eval surface and
    29.G post-training validation.
    """
    arr_x = np.asarray(list(x), dtype=np.float64)
    arr_y = np.asarray(list(y), dtype=np.float64)
    if arr_x.shape != arr_y.shape:
        raise ValueError(
            f"shape mismatch: x={arr_x.shape}, y={arr_y.shape}"
        )
    if arr_x.size < 2:
        return RankCorrelation(rho=0.0, n=int(arr_x.size))
    rx = _rankdata(arr_x)
    ry = _rankdata(arr_y)
    if rx.std() == 0 or ry.std() == 0:
        return RankCorrelation(rho=0.0, n=int(arr_x.size))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return RankCorrelation(rho=rho, n=int(arr_x.size))


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error between two equal-length sequences."""
    arr_x = np.asarray(list(x), dtype=np.float64)
    arr_y = np.asarray(list(y), dtype=np.float64)
    if arr_x.shape != arr_y.shape:
        raise ValueError(
            f"shape mismatch: x={arr_x.shape}, y={arr_y.shape}"
        )
    if arr_x.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr_x - arr_y)))


def bias(predicted: Sequence[float], target: Sequence[float]) -> float:
    """Mean signed error ``mean(predicted − target)`` — direction of
    miscalibration. Positive bias means the student over-predicts."""
    arr_p = np.asarray(list(predicted), dtype=np.float64)
    arr_t = np.asarray(list(target), dtype=np.float64)
    if arr_p.shape != arr_t.shape:
        raise ValueError(
            f"shape mismatch: predicted={arr_p.shape}, target={arr_t.shape}"
        )
    if arr_p.size == 0:
        return 0.0
    return float(np.mean(arr_p - arr_t))


def empirical_coverage(
    predicted: Sequence[float],
    target: Sequence[float],
    *,
    quantile: float,
) -> float:
    """Fraction of targets falling inside ``[predicted ± quantile]``,
    clipped to ``[0, 1]``. The D10-A coverage diagnostic."""
    if quantile < 0:
        raise ValueError(f"quantile must be non-negative; got {quantile}")
    arr_p = np.asarray(list(predicted), dtype=np.float64)
    arr_t = np.asarray(list(target), dtype=np.float64)
    if arr_p.shape != arr_t.shape:
        raise ValueError(
            f"shape mismatch: predicted={arr_p.shape}, target={arr_t.shape}"
        )
    if arr_p.size == 0:
        return 0.0
    low = np.maximum(0.0, arr_p - quantile)
    high = np.minimum(1.0, arr_p + quantile)
    return float(np.mean((arr_t >= low) & (arr_t <= high)))


def calibration_drift(empirical: float, target: float) -> float:
    """Signed drift ``empirical - target``; negative means under-coverage."""
    return float(empirical) - float(target)


def memorisation_gap(train_rho: float, test_rho: float) -> float:
    """``train_rho - test_rho`` per :doc:`PHASE_29_PLAN.md` §5 D7 pass
    criterion 2 ("memorisation gap ≤ 0.10")."""
    return float(train_rho) - float(test_rho)


def passes_spearman_floor(
    per_env_rho: Sequence[float], *, floor: float = 0.70
) -> bool:
    """Pass criterion 1: every held-out env Spearman ρ above ``floor``."""
    return all(float(r) >= floor for r in per_env_rho)


def passes_calibration_drift(drift: float, *, tol: float = 0.05) -> bool:
    """Pass criterion 4: |empirical - target| ≤ ``tol`` (default 5 pp)."""
    return abs(float(drift)) <= float(tol)


def passes_rewardbench(score: float, *, floor: float = 0.65) -> bool:
    """Pass criterion 3: RewardBench overall ≥ ``floor`` (default 0.65)."""
    return float(score) >= float(floor)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank ranking (mirrors scipy ``rankdata(method="average")``)."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


__all__ = [
    "RankCorrelation",
    "bias",
    "calibration_drift",
    "empirical_coverage",
    "mae",
    "memorisation_gap",
    "passes_calibration_drift",
    "passes_rewardbench",
    "passes_spearman_floor",
    "spearman_rho",
]
