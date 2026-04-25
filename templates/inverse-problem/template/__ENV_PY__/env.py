"""__ENV_CLASS__ — top-level RL environment handle for __ENV_ID__.

Mirrors the interface used by every shipped env in
``verifiable_labs_envs.envs``: ``generate_instance(seed)`` for
ground-truth + measurement, ``score(prediction, instance)`` for the
reward dict, and ``run_baseline(seed)`` as the convenience round-trip.

The first instantiation of the env runs ``calibrate_quantile`` to
fix the conformal threshold ``q_α``; the result is cached in-memory
for the process lifetime. For tests, pass
``calibration_quantile=...`` explicitly to skip calibration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from __ENV_PY__.data import Instance, Prediction, generate_ground_truth
from __ENV_PY__.forward_op import forward
from __ENV_PY__.reward import DEFAULT_ALPHA, DEFAULT_WEIGHTS, compute_reward

NAME = "__ENV_ID__"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "noise_sigma": 0.05,
    # TODO: domain-specific hyperparams (resolution, k, m, ...).
}


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    """Sample a fresh ground-truth, apply the forward operator, add noise."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    rng = np.random.default_rng(seed)
    x_true = generate_ground_truth(seed, **params)
    y_clean = forward(x_true, seed=seed)
    noise_sigma = float(params["noise_sigma"])
    noise = noise_sigma * rng.standard_normal(y_clean.shape)
    y = y_clean + noise
    return Instance(
        y=y,
        x_true=x_true,
        seed=int(seed),
        metadata={"noise_sigma": noise_sigma, "alpha": float(params["alpha"])},
    )


def baseline_predict(instance: Instance) -> Prediction:
    """Reference solver — used by calibration + ``run_baseline``.

    TODO: replace this stub with your domain's classical baseline
    (OMP / FBP / Gerchberg-Saxton / zero-fill / ...). The default
    returns ``x_hat = 0``, which is intentionally weak so calibration
    on the scaffold isn't accidentally tight.
    """
    n_true = instance.x_true.size
    return Prediction(
        x_hat=np.zeros_like(instance.x_true),
        sigma_hat=np.ones_like(instance.x_true) * 0.5,
    )


class __ENV_CLASS__:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(self, prediction: Prediction, instance: Instance) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            conformal_quantile=self.conformal_quantile,
            weights=self.weights,
            alpha=float(instance.metadata.get("alpha", DEFAULT_ALPHA)),
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)


def calibrate_quantile(n_samples: int = 100, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute the ``(1-α)`` quantile of baseline residuals over
    ``n_samples`` fresh seeds. Used to fix the conformal width."""
    nc_scores: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        sigma = np.maximum(np.asarray(pred.sigma_hat), 1e-9)
        residual = np.abs(np.asarray(inst.x_true) - np.asarray(pred.x_hat)) / sigma
        nc_scores.append(float(residual.max()))
    arr = np.asarray(nc_scores)
    return float(np.quantile(arr, 1.0 - alpha))


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> __ENV_CLASS__:
    """Factory mirroring the verifiers convention. Pass
    ``calibration_quantile`` to skip auto-calibration in tests."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 30 if fast else 200
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return __ENV_CLASS__(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "Instance",
    "Prediction",
    "__ENV_CLASS__",
    "generate_instance",
    "baseline_predict",
    "calibrate_quantile",
    "load_environment",
]
