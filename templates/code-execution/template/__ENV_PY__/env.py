"""__ENV_CLASS__ — top-level RL environment handle for __ENV_ID__.

Mirrors the interface used by every shipped env in
``verifiable_labs_envs.envs``: ``generate_instance(seed)`` for the
prompt + hidden gold solution, ``score(prediction, instance)`` for
the reward dict, and ``run_baseline(seed)`` as the convenience
round-trip.

The first instantiation of the env runs ``calibrate_quantile`` to
fix the conformal threshold ``q̂_α`` against a held-out set of
baseline predictions; the result is cached in-memory for the
process lifetime. For tests, pass ``calibration_quantile=...``
explicitly to skip calibration.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from __ENV_PY__.data import CodeInstance, CodePrediction, generate_problem
from __ENV_PY__.reward import (
    DEFAULT_ALPHA,
    DEFAULT_TIMEOUT_S_PER_CALL,
    DEFAULT_WEIGHTS,
    compute_reward,
)
from __ENV_PY__.sandbox import DEFAULT_MEM_BYTES

NAME = "__ENV_ID__"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "sandbox_timeout_s": DEFAULT_TIMEOUT_S_PER_CALL,
    "sandbox_mem_bytes": DEFAULT_MEM_BYTES,
    # TODO: domain-specific hyperparams (max_helpers, num_visible_tests, ...).
}


def generate_instance(seed: int, **kwargs: Any) -> CodeInstance:
    """Sample a fresh problem and wrap it in a :class:`CodeInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    return CodeInstance(
        function_signature=problem["function_signature"],
        docstring=problem["docstring"],
        visible_tests=tuple(problem["visible_tests"]),
        hidden_tests=tuple(problem["hidden_tests"]),
        gold_solution=problem["gold_solution"],
        template_name=problem["template_name"],
        seed=int(seed),
        metadata={
            "alpha": float(params["alpha"]),
            "sandbox_timeout_s": float(params["sandbox_timeout_s"]),
        },
    )


def baseline_predict(instance: CodeInstance) -> CodePrediction:
    """Reference solver — used by calibration + ``run_baseline``.

    The default returns an empty prediction (``code=""``,
    ``confidence=0``), which scores
    ``format_valid=parse_valid=pass_rate=0`` so calibration on the
    scaffold isn't accidentally tight. TODO: replace with a
    domain-appropriate weak baseline (e.g. ``def f(*a, **k): return 0``)
    so the conformal quantile lands in a useful range.
    """
    del instance  # unused in default baseline
    return CodePrediction(code="", raw="", confidence=0.0)


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

    def generate_instance(self, seed: int, **kwargs: Any) -> CodeInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: CodePrediction,
        instance: CodeInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            timeout_s=float(self.hyperparams["sandbox_timeout_s"]),
            mem_bytes=int(self.hyperparams["sandbox_mem_bytes"]),
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
        # Calibration is expensive (each seed spawns a sandboxed
        # pytest); ``fast=True`` keeps test-suite invocations cheap.
        n = 5 if fast else 30
        q = _cached_quantile(n, DEFAULT_ALPHA)
    return __ENV_CLASS__(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "CodeInstance",
    "CodePrediction",
    "__ENV_CLASS__",
    "generate_instance",
    "baseline_predict",
    "calibrate_quantile",
    "load_environment",
]
