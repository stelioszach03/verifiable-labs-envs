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
from __ENV_PY__.data import SqlInstance, SqlPrediction, generate_problem
from __ENV_PY__.reward import (
    DEFAULT_ALPHA,
    DEFAULT_WEIGHTS,
    compute_reward,
)
from __ENV_PY__.sandbox import (
    DEFAULT_MAX_QUERY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    Schema,
)

NAME = "__ENV_ID__"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "alpha": DEFAULT_ALPHA,
    "max_rows": DEFAULT_MAX_ROWS,
    "timeout_s": DEFAULT_TIMEOUT_S,
    "max_query_bytes": DEFAULT_MAX_QUERY_BYTES,
}


def generate_instance(seed: int, **kwargs: Any) -> SqlInstance:
    """Sample a fresh problem and wrap it in a :class:`SqlInstance`."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    problem = generate_problem(int(seed), **params)
    schema = Schema(
        create_statements=tuple(problem["create_statements"]),
        seed_statements=tuple(problem["seed_statements"]),
        table_names=tuple(problem["table_names"]),
        column_names_by_table=dict(problem["column_names"]),
        seed=int(seed),
    )
    return SqlInstance(
        prompt=problem["prompt"],
        template_name=problem["template_name"],
        seed=int(seed),
        schema=schema,
        gold_query=problem["gold_query"],
        gold_query_is_ordered=bool(problem["gold_query_is_ordered"]),
        gold_result_rows=tuple(tuple(r) for r in problem["gold_result_rows"]),
        metadata={
            "alpha": float(params["alpha"]),
            "max_rows": int(params["max_rows"]),
            "timeout_s": float(params["timeout_s"]),
        },
    )


def baseline_predict(instance: SqlInstance) -> SqlPrediction:
    """Reference solver — empty query.

    The default returns an empty prediction; calibration produces a
    wide residual distribution. TODO: replace with a domain-
    appropriate weak baseline if the conformal quantile lands too
    tight.
    """
    del instance
    return SqlPrediction(query="", raw="", confidence=0.0)


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

    def generate_instance(self, seed: int, **kwargs: Any) -> SqlInstance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: SqlPrediction,
        instance: SqlInstance,
    ) -> dict[str, Any]:
        return compute_reward(
            prediction=prediction,
            instance=instance,
            weights=self.weights,
            timeout_s=float(self.hyperparams["timeout_s"]),
            max_rows=int(self.hyperparams["max_rows"]),
            max_query_bytes=int(self.hyperparams["max_query_bytes"]),
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
    "SqlInstance",
    "SqlPrediction",
    "__ENV_CLASS__",
    "build_user_prompt",
    "baseline_predict",
    "calibrate_quantile",
    "generate_instance",
    "load_environment",
    "parse_response",
]
