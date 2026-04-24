"""Automated conformal calibration for any inverse-problem env.

For a given env + baseline solver, this runs ``n_calibration`` instances
through the baseline, scores them, and returns the ``1-alpha`` quantile of
the empirical non-conformity distribution — suitable as the
``conformal_quantile`` argument to the env factory.

This replaces the per-env hand-tuned ``_cached_quantile`` pattern in the
legacy envs. New envs should call ``auto_calibrate(env, baseline)``
once at module import.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalConfig:
    """Bundle of calibration outputs used by env.score."""

    quantile: float
    alpha: float
    n_calibration: int
    non_conformity_mean: float
    non_conformity_std: float


def _default_non_conformity(
    prediction: Any, instance: Any
) -> float:
    """Generic residual magnitude: ``||x_true - x_hat|| / sigma_hat``.

    Works for any env that exposes ``x_true`` on the instance and ``x_hat`` +
    ``sigma_hat`` on the prediction. Each env can override by passing its
    own ``non_conformity_fn``.
    """
    x_true = np.asarray(instance.x_true).ravel()
    x_hat = np.asarray(prediction.x_hat).ravel()
    sigma_hat = np.asarray(getattr(prediction, "sigma_hat", np.ones_like(x_hat))).ravel()
    sigma_hat = np.maximum(sigma_hat, 1e-9)
    return float(np.max(np.abs(x_true - x_hat) / sigma_hat))


def auto_calibrate(
    generate_instance: Callable[[int], Any],
    run_baseline: Callable[[Any], Any],
    *,
    n_calibration: int = 200,
    alpha: float = 0.1,
    non_conformity_fn: Callable[[Any, Any], float] | None = None,
    seed_start: int = 10_000,
) -> ConformalConfig:
    """Return a ``ConformalConfig`` for ``(1-alpha)`` coverage on ``generate_instance``.

    Parameters
    ----------
    generate_instance : Callable[[int], Instance]
        Given a seed, return a fresh instance. The env's
        ``generate_instance`` method usually fits.
    run_baseline : Callable[[Instance], Prediction]
        Runs the classical baseline on the instance and returns a
        Prediction-shaped object with ``x_hat`` and optionally ``sigma_hat``.
    n_calibration : int
        Number of calibration instances. 200 is a good default for
        stability while keeping calibration runtime reasonable.
    alpha : float
        Miscoverage level. The returned quantile is the ``1 - alpha``
        quantile of non-conformity scores.
    non_conformity_fn : Callable, optional
        Custom non-conformity metric. Defaults to the generic max-
        normalized-residual above.
    seed_start : int
        Starting seed for the calibration set. Kept large (10k) so
        calibration seeds don't overlap with training/eval seeds.

    Returns
    -------
    ConformalConfig
    """
    if n_calibration < 2:
        raise ValueError(f"n_calibration must be >= 2; got {n_calibration}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    nc_fn = non_conformity_fn or _default_non_conformity

    scores: list[float] = []
    for i in range(n_calibration):
        instance = generate_instance(seed_start + i)
        prediction = run_baseline(instance)
        scores.append(float(nc_fn(prediction, instance)))

    arr = np.asarray(scores, dtype=np.float64)
    q = float(np.quantile(arr, 1.0 - alpha))
    return ConformalConfig(
        quantile=q,
        alpha=float(alpha),
        n_calibration=int(n_calibration),
        non_conformity_mean=float(arr.mean()),
        non_conformity_std=float(arr.std()),
    )


__all__ = ["ConformalConfig", "auto_calibrate"]
