"""__ENV_CLASS__ — top-level RL environment handle for __ENV_ID__.

Mirrors the four-method contract used by every shipped env in
``verifiable_labs_envs.envs``: ``generate_instance(seed)``,
``score(prediction, instance)``, ``run_baseline(seed)``, and the
module-level ``load_environment(calibration_quantile=None,
fast=True)`` factory.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from __ENV_PY__.adapter import build_user_prompt, parse_response  # noqa: F401  (re-export)
from __ENV_PY__.corpus import (
    DEFAULT_DOCUMENT_COUNT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEST_TOKENS,
)
from __ENV_PY__.data import NeedleInstance, NeedlePrediction, generate_problem
from __ENV_PY__.reward import (
    DEFAULT_ALPHA,
    DEFAULT_WEIGHTS,
    compute_reward,
)

NAME = "__ENV_ID__"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "target_tokens": DEFAULT_TEST_TOKENS,
    "document_count": DEFAULT_DOCUMENT_COUNT,
    "max_tokens": DEFAULT_MAX_TOKENS,
}


def generate_instance(seed: int, **kwargs: Any) -> NeedleInstance:
    """Sample a fresh problem and wrap it in a :class:`NeedleInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return NeedleInstance(
        question=problem["question"],
        template_name=problem.get("template_name", "default"),
        seed=int(seed),
        corpus=problem["corpus"],
        needle_text=problem["needle_text"],
        needle_anchor=problem["needle_anchor"],
        position_mode=problem["position_mode"],
        metadata={
            "alpha": float(params["alpha"]),
            "target_tokens": int(params["target_tokens"]),
            "needle_token": problem.get("needle_token", problem["needle_text"]),
        },
    )


def baseline_predict(instance: NeedleInstance) -> NeedlePrediction:
    """Reference solver — empty answer.

    The default returns an empty prediction; calibration produces a
    wide residual distribution. TODO: replace with a domain-
    appropriate weak baseline if the conformal quantile lands too
    tight.
    """
    del instance
    return NeedlePrediction(answer="", raw="", confidence=0.0)


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

    def generate_instance(self, seed: int, **kwargs: Any) -> NeedleInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: NeedlePrediction,
        instance: NeedleInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            conformal_quantile=self.conformal_quantile,
        )

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = baseline_predict(instance)
        return self.score(prediction, instance)


def calibrate_quantile(
    n_samples: int = 30,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Compute the ``(1 − α)`` quantile of baseline residuals."""
    import numpy as np  # noqa: PLC0415

    from verifiable_labs_envs.conformal import split_conformal_quantile  # noqa: PLC0415

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
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return __ENV_CLASS__(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "NeedleInstance",
    "NeedlePrediction",
    "__ENV_CLASS__",
    "build_user_prompt",
    "baseline_predict",
    "calibrate_quantile",
    "generate_instance",
    "load_environment",
    "parse_response",
]
