"""__ENV_CLASS__ — top-level RL environment handle for __ENV_ID__.

Mirrors the four-method contract used by every shipped env in
``verifiable_labs_envs.envs``: ``generate_instance(seed)``,
``score(prediction, instance)``, ``run_baseline(seed)``, and the
module-level ``load_environment(calibration_quantile=None,
fast=True)`` factory.

The first instantiation runs ``calibrate_quantile`` to fix the
conformal threshold ``q̂_α`` against a held-out set of baseline
predictions; the result is cached in-memory for the process
lifetime. For tests, pass ``calibration_quantile=...`` explicitly
to skip calibration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from __ENV_PY__.adapter import build_user_prompt, parse_response  # noqa: F401  (re-export)
from __ENV_PY__.data import ToolCallingInstance, ToolCallingPrediction, generate_problem
from __ENV_PY__.reward import (
    DEFAULT_ALPHA,
    DEFAULT_WEIGHTS,
    compute_reward,
)
from __ENV_PY__.tools import init_state

NAME = "__ENV_ID__"

DEFAULT_MAX_TOOL_CALLS: int = 30

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
}


def generate_instance(seed: int, **kwargs: Any) -> ToolCallingInstance:
    """Sample a fresh problem and wrap it in a :class:`ToolCallingInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return ToolCallingInstance(
        prompt=problem["prompt"],
        template_name=problem["template_name"],
        seed=int(seed),
        gold_spec=dict(problem["gold_spec"]),
        initial_files=dict(problem["initial_files"]),
        available_tools=tuple(problem["available_tools"]),
        metadata={
            "alpha": float(params["alpha"]),
            "max_tool_calls": int(params["max_tool_calls"]),
        },
    )


def baseline_predict(instance: ToolCallingInstance) -> ToolCallingPrediction:
    """Reference solver — empty trajectory, empty submission.

    Empty trajectory scores zero on every component; the wide
    residual distribution this produces yields a non-trivial
    conformal quantile when calibration runs over a baseline sweep.
    """
    return ToolCallingPrediction(
        tool_calls=(),
        final_text="",
        final_state=init_state(seed=instance.seed, initial_files=instance.initial_files),
        raw="",
        confidence=0.0,
    )


class __ENV_CLASS__:
    """RL environment handle wrapping one calibrated conformal quantile."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0; got {max_tool_calls}")
        self.max_tool_calls = int(max_tool_calls)
        self.env_id: str = ""
        self.env_args: dict[str, Any] = {}

    def generate_instance(self, seed: int, **kwargs: Any) -> ToolCallingInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: ToolCallingPrediction,
        instance: ToolCallingInstance,
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
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> __ENV_CLASS__:
    """Factory mirroring the verifiers convention."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return __ENV_CLASS__(conformal_quantile=q, max_tool_calls=max_tool_calls)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_MAX_TOOL_CALLS",
    "ToolCallingInstance",
    "ToolCallingPrediction",
    "__ENV_CLASS__",
    "build_user_prompt",
    "baseline_predict",
    "calibrate_quantile",
    "generate_instance",
    "load_environment",
    "parse_response",
]
