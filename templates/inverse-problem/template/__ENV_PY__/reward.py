"""Conformal-calibrated reward for __ENV_ID__.

The reward function combines a domain-appropriate point-estimate
metric (NMSE / SSIM / log-likelihood / ...) with a split-conformal
coverage term that rewards honest per-entry uncertainty. Both
ingredients are weighted into a single scalar in ``[0, 1]``.

The conformal quantile ``q_α`` is calibrated once at env construction
time over a held-out set of baseline runs; ``compute_reward`` uses it
to evaluate empirical coverage of the predicted interval against the
target ``1 - α``.

TODO: replace ``score_point_estimate`` and (optionally) the weights
to fit your domain. Default uses NMSE-by-exp on flat real signals.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from __ENV_PY__.data import Instance, Prediction

DEFAULT_ALPHA: float = 0.1
DEFAULT_NMSE_TAU: float = 0.5
DEFAULT_WEIGHTS: dict[str, float] = {"point": 0.7, "conformal": 0.3}


def score_point_estimate(prediction: Prediction, instance: Instance) -> float:
    """Score the point estimate in ``[0, 1]``. Default: ``exp(-NMSE/τ)``.

    TODO: swap this for SSIM / PSNR / per-residue accuracy / etc. as
    appropriate for your domain.
    """
    x_true = np.asarray(instance.x_true).ravel()
    x_hat = np.asarray(prediction.x_hat).ravel()
    denom = max(float(np.sum(x_true ** 2)), 1e-12)
    nmse = float(np.sum((x_true - x_hat) ** 2)) / denom
    return float(np.exp(-nmse / DEFAULT_NMSE_TAU))


def empirical_coverage(
    prediction: Prediction,
    instance: Instance,
    conformal_quantile: float,
) -> float:
    """Fraction of true entries inside the conformal interval."""
    x_true = np.asarray(instance.x_true).ravel()
    x_hat = np.asarray(prediction.x_hat).ravel()
    sigma = np.asarray(prediction.sigma_hat).ravel()
    width = conformal_quantile * sigma
    inside = (x_true >= x_hat - width) & (x_true <= x_hat + width)
    return float(np.mean(inside)) if inside.size else 0.0


def coverage_score(empirical: float, target: float) -> float:
    """Reward peaks at ``empirical = target``; symmetric falloff in both
    directions so over- and under-coverage are penalised equally."""
    return float(max(0.0, 1.0 - abs(empirical - target) / max(target, 1e-9)))


def compute_reward(
    prediction: Prediction,
    instance: Instance,
    conformal_quantile: float,
    *,
    weights: dict[str, float] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Combine point + conformal scores into the env reward dict."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    score_point = score_point_estimate(prediction, instance)
    cov = empirical_coverage(prediction, instance, conformal_quantile)
    target = 1.0 - alpha
    score_conf = coverage_score(cov, target)
    reward = w["point"] * score_point + w["conformal"] * score_conf
    return {
        "reward": float(reward),
        "components": {
            "point": float(score_point),
            "conformal": float(score_conf),
        },
        "meta": {
            "coverage": cov,
            "target_coverage": target,
            "conformal_quantile": float(conformal_quantile),
            "weights": dict(w),
        },
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_NMSE_TAU",
    "DEFAULT_WEIGHTS",
    "score_point_estimate",
    "empirical_coverage",
    "coverage_score",
    "compute_reward",
]
