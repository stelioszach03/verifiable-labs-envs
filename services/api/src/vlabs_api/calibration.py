"""Triple-based calibration helpers — wraps the ``vlabs-calibrate`` SDK.

The hosted API never sees a customer reward function; it operates only on
``(predicted_reward, reference_reward, uncertainty)`` triples. These
helpers translate that shape into the ``vlabs_calibrate`` non-conformity
machinery (``score_fn``, ``interval_fn``).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from vlabs_calibrate import core
from vlabs_calibrate.nonconformity import NonconformityScore
from vlabs_calibrate.nonconformity import get as get_nonconformity

from vlabs_api.errors import (
    InvalidScore,
    InvalidUncertainty,
    MissingRequiredKeys,
    UnknownNonconformity,
)
from vlabs_api.schemas import CalibrationTrace


def resolve_nonconformity(name: str) -> NonconformityScore:
    try:
        return get_nonconformity(name)
    except ValueError as exc:
        raise UnknownNonconformity(detail=str(exc)) from exc


class CalibrationOutcome(NamedTuple):
    quantile: float
    nonconformity_stats: dict[str, float]


def _trace_to_dict(trace: CalibrationTrace) -> dict[str, float]:
    out: dict[str, float] = {"reference_reward": float(trace.reference_reward)}
    if trace.uncertainty is not None:
        out["uncertainty"] = float(trace.uncertainty)
    return out


def _validate_required_keys(
    nc: NonconformityScore, traces: list[CalibrationTrace], context: str
) -> None:
    for i, t in enumerate(traces):
        if "uncertainty" in nc.required_trace_keys and t.uncertainty is None:
            raise MissingRequiredKeys(
                detail=(
                    f"{context}: trace[{i}] requires 'uncertainty' for "
                    f"nonconformity={nc.name!r}"
                )
            )
        if "uncertainty" in nc.required_trace_keys and (
            t.uncertainty is not None and t.uncertainty < 0
        ):
            raise InvalidUncertainty(
                detail=f"{context}: trace[{i}].uncertainty={t.uncertainty} is negative"
            )


def calibrate_from_triples(
    traces: list[CalibrationTrace],
    alpha: float,
    nonconformity_name: str,
    eps: float = 1e-8,
) -> CalibrationOutcome:
    """Compute the conformal quantile + non-conformity stats from triples."""
    nc = resolve_nonconformity(nonconformity_name)
    _validate_required_keys(nc, traces, context="calibrate")

    scores = np.empty(len(traces), dtype=np.float64)
    for i, trace in enumerate(traces):
        score = nc.score_fn(_trace_to_dict(trace), float(trace.predicted_reward), eps)
        if not math.isfinite(score):
            raise InvalidScore(
                detail=(
                    f"calibrate: trace[{i}] produced non-finite score {score!r} for "
                    f"nonconformity={nonconformity_name!r}"
                )
            )
        scores[i] = score

    quantile = core.split_conformal_quantile(scores, alpha=alpha)
    stats = {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "median": float(np.median(scores)),
    }
    return CalibrationOutcome(quantile=quantile, nonconformity_stats=stats)


def predict_interval(
    nonconformity_name: str,
    quantile: float,
    predicted_reward: float,
    sigma: float | None,
) -> tuple[float, float, float]:
    """Return ``(lower, upper, sigma_used)`` for a single prediction."""
    nc = resolve_nonconformity(nonconformity_name)
    if nc.is_scale_aware:
        if sigma is None:
            raise MissingRequiredKeys(
                detail=(
                    f"predict: nonconformity={nonconformity_name!r} is scale-aware; "
                    f"'uncertainty' (sigma) is required"
                )
            )
        if sigma < 0:
            raise InvalidUncertainty(detail=f"predict: sigma={sigma} is negative")
        sigma_used = float(sigma)
    else:
        sigma_used = float(sigma) if sigma is not None else 0.0

    lower, upper = nc.interval_fn(float(predicted_reward), sigma_used, float(quantile))
    return float(lower), float(upper), sigma_used


def evaluate_against_calibration(
    traces: list[CalibrationTrace],
    nonconformity_name: str,
    quantile: float,
    alpha: float,
    tolerance: float,
    eps: float = 1e-8,
) -> dict[str, float | int | bool | dict[str, float]]:
    """Run a held-out evaluation; return a CoverageReport-shaped dict."""
    nc = resolve_nonconformity(nonconformity_name)
    _validate_required_keys(nc, traces, context="evaluate")

    widths: list[float] = []
    scores = np.empty(len(traces), dtype=np.float64)
    n_in = 0
    for i, trace in enumerate(traces):
        sigma_val = (
            float(trace.uncertainty) if (trace.uncertainty is not None and nc.is_scale_aware) else 0.0
        )
        predicted = float(trace.predicted_reward)
        lower, upper = nc.interval_fn(predicted, sigma_val, float(quantile))
        if lower <= float(trace.reference_reward) <= upper:
            n_in += 1
        widths.append(float(upper - lower))
        score = nc.score_fn(_trace_to_dict(trace), predicted, eps)
        if not math.isfinite(score):
            raise InvalidScore(
                detail=f"evaluate: trace[{i}] produced non-finite score {score!r}"
            )
        scores[i] = score

    n = len(traces)
    target = 1.0 - alpha
    empirical = n_in / n
    return {
        "target_coverage": target,
        "empirical_coverage": empirical,
        "n": n,
        "n_in_interval": n_in,
        "interval_width_mean": float(np.mean(widths)),
        "interval_width_median": float(np.median(widths)),
        "tolerance": float(tolerance),
        "passes": bool(abs(empirical - target) <= tolerance),
        "nonconformity": {
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "median": float(np.median(scores)),
        },
    }


__all__ = [
    "CalibrationOutcome",
    "calibrate_from_triples",
    "predict_interval",
    "evaluate_against_calibration",
    "resolve_nonconformity",
]
