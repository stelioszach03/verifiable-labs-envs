"""__ENV_CLASS__ — top-level RL environment handle for __ENV_ID__.

Mirrors the interface used by every shipped env in
``verifiable_labs_envs.envs``: ``generate_instance(seed)`` for the
problem prompt + hidden gold expression, ``score(prediction,
instance)`` for the reward dict, and ``run_baseline(seed)`` as the
convenience round-trip.

The first instantiation of the env runs ``calibrate_quantile`` to fix
the conformal threshold ``q̂_α`` against a held-out set of baseline
predictions; the result is cached in-memory for the process lifetime.
For tests, pass ``calibration_quantile=...`` explicitly to skip
calibration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from __ENV_PY__.data import Instance, Prediction, generate_problem
from __ENV_PY__.reward import (
    DEFAULT_ALPHA,
    DEFAULT_SIMPLIFY_TIMEOUT_S,
    DEFAULT_WEIGHTS,
    compute_reward,
)

NAME = "__ENV_ID__"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "simplify_timeout_s": DEFAULT_SIMPLIFY_TIMEOUT_S,
    # TODO: domain-specific hyperparams (max_depth, max_terms, ...).
}


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    """Sample a fresh ``(prompt, gold_expr)`` pair as an :class:`Instance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    prompt, gold_expr = generate_problem(seed, **params)
    return Instance(
        prompt=prompt,
        gold_expr=gold_expr,
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "simplify_timeout_s": float(params["simplify_timeout_s"]),
        },
    )


def baseline_predict(instance: Instance) -> Prediction:
    """Reference solver — used by calibration + ``run_baseline``.

    The default returns an empty prediction (``answer_expr=""``,
    ``confidence=0``), which scores ``format_valid=parse_valid=correct=0``
    so calibration on the scaffold isn't accidentally tight. TODO:
    replace this with a domain-appropriate weak baseline (e.g. random
    monomial guess for math-algebra) so the conformal quantile lands
    in a useful range.
    """
    del instance  # unused in default baseline
    return Prediction(answer_expr="", raw="", confidence=0.0)


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
            weights=self.weights,
            timeout_s=float(self.hyperparams["simplify_timeout_s"]),
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)


def calibrate_quantile(n_samples: int = 100, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals over
    ``n_samples`` fresh seeds. Reuses
    ``verifiable_labs_envs.conformal.split_conformal_quantile`` for
    the finite-sample correction."""
    import numpy as np  # noqa: PLC0415

    from verifiable_labs_envs.conformal import split_conformal_quantile

    residuals: list[float] = []
    for seed in range(n_samples):
        inst = generate_instance(seed)
        pred = baseline_predict(inst)
        out = compute_reward(prediction=pred, instance=inst)
        residuals.append(1.0 - float(out["reward"]))
    return float(split_conformal_quantile(np.asarray(residuals), alpha))


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
