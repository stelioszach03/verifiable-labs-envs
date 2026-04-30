"""Built-in non-conformity scores and the registration mechanism.

A non-conformity score has two sides:

1. **At calibration time** — given a trace dict and the predicted reward,
   return a non-negative scalar score. Larger means "more anomalous".
2. **At test time** — given the predicted reward, σ, and the calibrated
   quantile ``q``, return a ``(lower, upper)`` interval. Scale-aware scores
   use σ; scale-free scores ignore it.

The two pieces are bundled in :class:`NonconformityScore`. Three built-ins
are registered: :data:`SCALED_RESIDUAL` (default), :data:`ABS_RESIDUAL`,
and :data:`BINARY`.

User-supplied callables ``f(trace, predicted_reward) -> float`` are also
accepted by :func:`vlabs_calibrate.calibrate` — they are wrapped as a
scale-aware score with a ``[predicted ± q · σ]`` interval at test time.
For full control over the interval shape, build a :class:`NonconformityScore`
directly and register it (or pass it as the ``nonconformity`` argument).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

ScoreFn = Callable[[Mapping[str, Any], float, float], float]
IntervalFn = Callable[[float, float, float], tuple[float, float]]


@dataclass(frozen=True)
class NonconformityScore:
    """A pair of (score, interval) functions plus metadata.

    Parameters
    ----------
    name
        Identifier — used in ``CalibrationResult.nonconformity_name`` and
        in error messages. Must be a valid Python identifier for registry use.
    required_trace_keys
        Keys every calibration / evaluation trace MUST contain. Validated
        once at calibration time and once per ``evaluate`` call.
    score_fn
        ``score_fn(trace, predicted_reward, eps) -> non-negative float``.
    interval_fn
        ``interval_fn(predicted_reward, sigma, quantile) -> (lower, upper)``.
    is_scale_aware
        ``True`` if ``interval_fn`` uses ``sigma``; informational, used by
        :class:`vlabs_calibrate.CalibratedRewardFn` to decide whether the
        caller needs to supply a ``sigma=`` argument.
    """

    name: str
    required_trace_keys: tuple[str, ...]
    score_fn: ScoreFn
    interval_fn: IntervalFn
    is_scale_aware: bool


def _scaled_residual_score(
    trace: Mapping[str, Any], predicted: float, eps: float
) -> float:
    sigma = float(trace["uncertainty"])
    return float(abs(predicted - float(trace["reference_reward"])) / max(sigma, eps))


def _scaled_residual_interval(
    predicted: float, sigma: float, quantile: float
) -> tuple[float, float]:
    delta = quantile * sigma
    return predicted - delta, predicted + delta


SCALED_RESIDUAL = NonconformityScore(
    name="scaled_residual",
    required_trace_keys=("reference_reward", "uncertainty"),
    score_fn=_scaled_residual_score,
    interval_fn=_scaled_residual_interval,
    is_scale_aware=True,
)
"""Default. ``|reward − reference| / max(σ, eps)``.

Use when each trace carries a per-sample uncertainty ``σ``. The interval
width scales with ``σ`` at test time, which is the standard split-conformal
behaviour and is appropriate for continuous rewards with calibrated σ
estimates (e.g. ensemble-disagreement, judge confidence, or model logprob).
"""


def _abs_residual_score(
    trace: Mapping[str, Any], predicted: float, eps: float  # noqa: ARG001
) -> float:
    return float(abs(predicted - float(trace["reference_reward"])))


def _abs_residual_interval(
    predicted: float, sigma: float, quantile: float  # noqa: ARG001
) -> tuple[float, float]:
    return predicted - quantile, predicted + quantile


ABS_RESIDUAL = NonconformityScore(
    name="abs_residual",
    required_trace_keys=("reference_reward",),
    score_fn=_abs_residual_score,
    interval_fn=_abs_residual_interval,
    is_scale_aware=False,
)
"""``|reward − reference|``, ignoring σ.

Use when no per-sample uncertainty is available. The interval width is a
constant ``q`` learned at calibration time; this is the marginal-coverage
version of split-conformal regression without scaling.
"""


def _binary_score(
    trace: Mapping[str, Any], predicted: float, eps: float  # noqa: ARG001
) -> float:
    return 0.0 if predicted == float(trace["reference_reward"]) else 1.0


def _binary_interval(
    predicted: float, sigma: float, quantile: float  # noqa: ARG001
) -> tuple[float, float]:
    # If quantile < 1.0 the calibration set agrees often enough that the
    # standard split-conformal guarantee says "predicted reward is correct" —
    # interval collapses to a point. Otherwise the guarantee is vacuous and
    # the interval covers the whole 0/1 range.
    if quantile < 1.0:
        return predicted, predicted
    return 0.0, 1.0


BINARY = NonconformityScore(
    name="binary",
    required_trace_keys=("reference_reward",),
    score_fn=_binary_score,
    interval_fn=_binary_interval,
    is_scale_aware=False,
)
"""``0.0 if predicted == reference else 1.0``.

Use for 0/1 rewards (HumanEval pass/fail, MATH exact-match, etc.).

.. warning::
    For binary rewards the standard split-conformal guarantee is degenerate:
    the ``(1 − α)`` quantile is either ``0`` (collapsing the interval to a
    point — the predicted reward is "verified") or ``1`` (yielding the
    vacuous ``[0, 1]`` interval). For non-trivial conditional coverage on
    binary tasks consider Mondrian / class-conditional conformal prediction
    (Vovk & Gammerman) — planned for ``0.2.0``.
"""


_REGISTRY: dict[str, NonconformityScore] = {
    "scaled_residual": SCALED_RESIDUAL,
    "abs_residual": ABS_RESIDUAL,
    "binary": BINARY,
}


def get(name: str) -> NonconformityScore:
    """Look up a registered non-conformity score by name.

    Raises
    ------
    ValueError
        If ``name`` is not registered. The error message lists the known
        registered names.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown nonconformity {name!r}; registered: {known}")
    return _REGISTRY[name]


def register(score: NonconformityScore, *, overwrite: bool = False) -> None:
    """Register a custom :class:`NonconformityScore` under its ``name``.

    Parameters
    ----------
    score
        The score object to register.
    overwrite
        If ``False`` (default), raise ``ValueError`` when a score with the
        same name is already registered. If ``True``, replace it.
    """
    if not overwrite and score.name in _REGISTRY:
        raise ValueError(
            f"nonconformity {score.name!r} already registered; "
            f"pass overwrite=True to replace"
        )
    _REGISTRY[score.name] = score


def registered_names() -> tuple[str, ...]:
    """Return the names of all currently-registered non-conformity scores."""
    return tuple(sorted(_REGISTRY))


__all__ = [
    "NonconformityScore",
    "SCALED_RESIDUAL",
    "ABS_RESIDUAL",
    "BINARY",
    "get",
    "register",
    "registered_names",
]
