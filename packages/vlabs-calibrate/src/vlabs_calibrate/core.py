"""Split-conformal prediction primitives.

These five functions are the numerical kernel of ``vlabs-calibrate``. Given
a calibration set of paired ``(x_hat_i, x_true_i, sigma_hat_i)`` tuples we
pool per-entry standardized absolute residuals
``r_i = |x_hat_i − x_true_i| / (sigma_hat_i + eps)`` and take the
``(1 − alpha)`` empirical quantile ``q_alpha`` with the finite-sample
correction ``ceil((n + 1) * (1 − alpha)) / n``. At test time, the
standardized-conformal interval ``[x_hat − q_alpha * sigma_hat,
x_hat + q_alpha * sigma_hat]`` has marginal coverage ``≥ 1 − alpha``
under exchangeability (Lei et al., 2018).

The reward for a predictor is maximized when its empirical coverage on
held-out instances matches the target ``1 − alpha`` — punishing both
overconfident (too-narrow) and overly-conservative (too-wide) uncertainty
estimates.

These primitives are deliberately env-agnostic: they accept plain NumPy
arrays and have no dependency on any reward-function or trace schema.
The high-level :func:`vlabs_calibrate.calibrate` API builds on them.
"""
from __future__ import annotations

import numpy as np


def split_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """``(1 − alpha)`` empirical quantile with the finite-sample split-conformal correction.

    For ``n`` calibration residuals this returns the
    ``ceil((n + 1) * (1 − alpha)) / n`` empirical quantile (clipped to 1.0),
    which gives the standard marginal-coverage guarantee in Lei et al. (2018).
    """
    if residuals.size == 0:
        raise ValueError("residuals must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n = residuals.size
    q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, q_level, method="higher"))


def scaled_residuals(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    sigma_hat: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-entry standardized absolute residuals ``|x_hat − x_true| / (sigma_hat + eps)``."""
    if x_hat.shape != x_true.shape or x_hat.shape != sigma_hat.shape:
        raise ValueError(
            f"shape mismatch: x_hat {x_hat.shape}, x_true {x_true.shape}, "
            f"sigma_hat {sigma_hat.shape}"
        )
    if np.any(sigma_hat < 0):
        raise ValueError("sigma_hat must be non-negative")
    return np.abs(x_hat - x_true) / (sigma_hat + eps)


def interval(
    x_hat: np.ndarray,
    sigma_hat: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized conformal interval ``[x_hat − q * sigma_hat, x_hat + q * sigma_hat]``."""
    if q < 0:
        raise ValueError(f"q must be non-negative; got {q}")
    lower = x_hat - q * sigma_hat
    upper = x_hat + q * sigma_hat
    return lower, upper


def coverage(x_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage — fraction of entries with ``x_true`` inside ``[lower, upper]``."""
    if x_true.shape != lower.shape or x_true.shape != upper.shape:
        raise ValueError(
            f"shape mismatch: x_true {x_true.shape}, lower {lower.shape}, upper {upper.shape}"
        )
    return float(np.mean((x_true >= lower) & (x_true <= upper)))


def coverage_score(coverage_observed: float, target: float) -> float:
    """Reward component that peaks at 1.0 when empirical coverage matches the target.

    Linear penalty on both over- and under-coverage, clipped to ``[0, 1]``.
    ``target`` and ``coverage_observed`` are both fractions in ``[0, 1]``.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be in [0, 1]; got {target}")
    return max(0.0, 1.0 - abs(coverage_observed - target))


__all__ = [
    "split_conformal_quantile",
    "scaled_residuals",
    "interval",
    "coverage",
    "coverage_score",
]
