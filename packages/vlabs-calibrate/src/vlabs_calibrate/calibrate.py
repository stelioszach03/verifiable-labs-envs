"""Public ``calibrate()`` entry point and the :class:`CalibratedRewardFn` wrapper.

This is the layer that turns the env-agnostic conformal primitives in
:mod:`vlabs_calibrate.core` into a one-line API for arbitrary reward
functions::

    import vlabs_calibrate as vc

    calibrated = vc.calibrate(my_reward, traces, alpha=0.1)
    result = calibrated(prompt=..., completion=..., ground_truth=..., sigma=0.5)

The :class:`CalibratedRewardFn` returned by :func:`calibrate` is a callable
drop-in for the original reward function — calling it returns a
:class:`vlabs_calibrate.types.CalibrationResult` with the raw reward plus
the conformal interval.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlabs_calibrate import core
from vlabs_calibrate.nonconformity import NonconformityScore
from vlabs_calibrate.nonconformity import get as _get_score
from vlabs_calibrate.types import CalibrationResult, CoverageReport

# Trace keys that are NEVER forwarded to the user's reward_fn.
_META_KEYS: frozenset[str] = frozenset({"reference_reward", "uncertainty", "_meta"})


def _resolve_nonconformity(
    nc: str | Callable[..., float] | NonconformityScore,
) -> tuple[NonconformityScore, str]:
    """Coerce the user-supplied ``nonconformity`` argument to a NonconformityScore."""
    if isinstance(nc, NonconformityScore):
        return nc, nc.name
    if isinstance(nc, str):
        return _get_score(nc), nc
    if callable(nc):
        # Bare callable: wrap with default scale-aware interval.
        def _custom_score(trace: Mapping[str, Any], predicted: float, eps: float) -> float:
            del eps  # eps not exposed to user-supplied callable
            return float(nc(trace, predicted))

        def _custom_interval(
            predicted: float, sigma: float, quantile: float
        ) -> tuple[float, float]:
            delta = quantile * sigma
            return predicted - delta, predicted + delta

        wrapped = NonconformityScore(
            name="<callable>",
            required_trace_keys=("reference_reward",),
            score_fn=_custom_score,
            interval_fn=_custom_interval,
            is_scale_aware=True,
        )
        return wrapped, "<callable>"
    raise TypeError(
        f"nonconformity must be a registered name (str), a callable "
        f"(trace, predicted) -> float, or a NonconformityScore; got {type(nc).__name__}"
    )


def _validate_trace_keys(
    trace: Mapping[str, Any],
    required: tuple[str, ...],
    *,
    index: int,
    context: str,
) -> None:
    missing = [k for k in required if k not in trace]
    if missing:
        raise ValueError(
            f"{context}: trace[{index}] is missing required key(s) {missing}; "
            f"got keys {sorted(trace.keys())}"
        )


def _extract_reward_kwargs(
    trace: Mapping[str, Any],
    keys: tuple[str, ...] | None,
) -> dict[str, Any]:
    if keys is None:
        return {k: v for k, v in trace.items() if k not in _META_KEYS}
    return {k: trace[k] for k in keys}


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "median": float(np.median(scores)),
    }


@dataclass(frozen=True)
class CalibratedRewardFn:
    """Callable wrapper around a reward function with conformal interval.

    Built by :func:`calibrate` — do not instantiate directly. The wrapper is
    a frozen dataclass so it is safe to share across processes (everything
    it carries is picklable as long as ``reward_fn`` is).
    """

    reward_fn: Callable[..., float]
    quantile: float
    alpha: float
    n_calibration: int
    nonconformity_name: str
    nonconformity_stats: Mapping[str, float]
    reward_kwargs_keys: tuple[str, ...] | None
    eps: float = 1e-8
    _nc_score: NonconformityScore = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._nc_score is None:
            raise ValueError(
                "CalibratedRewardFn must be constructed via vlabs_calibrate.calibrate(); "
                "the internal _nc_score field is required"
            )

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def is_scale_aware(self) -> bool:
        return self._nc_score.is_scale_aware

    def __call__(
        self,
        *args: Any,
        sigma: float | None = None,
        reference: float | None = None,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Run the wrapped ``reward_fn`` and return reward + conformal interval.

        Positional and keyword arguments are forwarded verbatim to the
        underlying ``reward_fn``. Two reserved keyword arguments are
        intercepted:

        * ``sigma`` — the per-call uncertainty. Required for scale-aware
          non-conformity scores; defaults to ``0.0`` for scale-free scores.
        * ``reference`` — optional held-out reference reward. When passed,
          ``CalibrationResult.covered`` is populated.
        """
        reward = float(self.reward_fn(*args, **kwargs))

        if self.is_scale_aware:
            if sigma is None:
                raise ValueError(
                    f"non-conformity {self.nonconformity_name!r} is scale-aware; "
                    f"pass sigma= to the calibrated reward function"
                )
            sigma_val = float(sigma)
            if sigma_val < 0:
                raise ValueError(f"sigma must be non-negative; got {sigma_val}")
        else:
            sigma_val = float(sigma) if sigma is not None else 0.0

        lower, upper = self._nc_score.interval_fn(reward, sigma_val, self.quantile)
        covered: bool | None = None
        if reference is not None:
            ref = float(reference)
            covered = bool(lower <= ref <= upper)

        return CalibrationResult(
            reward=reward,
            interval=(float(lower), float(upper)),
            sigma=sigma_val,
            quantile=self.quantile,
            alpha=self.alpha,
            target_coverage=self.target_coverage,
            covered=covered,
        )

    def evaluate(
        self,
        traces: Sequence[Mapping[str, Any]],
        *,
        tolerance: float = 0.05,
    ) -> CoverageReport:
        """Run on held-out traces; report empirical coverage + diagnostics.

        Parameters
        ----------
        traces
            Held-out evaluation set. Each trace must carry the same keys
            required at calibration time (in particular ``reference_reward``,
            and ``uncertainty`` for scale-aware scores).
        tolerance
            Absolute slack in coverage units used to set ``CoverageReport.passes``.
            ``passes`` is ``True`` iff ``|empirical − target| ≤ tolerance``.
            Default ``0.05`` (5 percentage points), matching the
            conformal-validation suite's threshold.

        Returns
        -------
        CoverageReport
        """
        if len(traces) == 0:
            raise ValueError("evaluate() requires at least one trace")
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative; got {tolerance}")

        required = self._nc_score.required_trace_keys
        widths: list[float] = []
        scores: list[float] = []
        n_in_interval = 0
        for i, trace in enumerate(traces):
            _validate_trace_keys(trace, required, index=i, context="evaluate")
            kwargs = _extract_reward_kwargs(trace, self.reward_kwargs_keys)
            predicted = float(self.reward_fn(**kwargs))
            sigma = float(trace["uncertainty"]) if self.is_scale_aware else 0.0
            lower, upper = self._nc_score.interval_fn(predicted, sigma, self.quantile)
            ref = float(trace["reference_reward"])
            if lower <= ref <= upper:
                n_in_interval += 1
            widths.append(float(upper - lower))
            scores.append(self._nc_score.score_fn(trace, predicted, self.eps))

        n = len(traces)
        empirical = n_in_interval / n
        target = self.target_coverage
        nc_stats = _summarize_scores(np.asarray(scores, dtype=np.float64))

        return CoverageReport(
            target_coverage=target,
            empirical_coverage=empirical,
            n=n,
            n_in_interval=n_in_interval,
            interval_width_mean=float(np.mean(widths)),
            interval_width_median=float(statistics.median(widths)),
            nonconformity=nc_stats,
            quantile=self.quantile,
            alpha=self.alpha,
            tolerance=float(tolerance),
            passes=bool(abs(empirical - target) <= tolerance),
        )


def calibrate(
    reward_fn: Callable[..., float],
    traces: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.1,
    nonconformity: str | Callable[..., float] | NonconformityScore = "scaled_residual",
    eps: float = 1e-8,
    reward_kwargs_keys: Sequence[str] | None = None,
) -> CalibratedRewardFn:
    """Wrap ``reward_fn`` with split-conformal coverage guarantees.

    Parameters
    ----------
    reward_fn
        Any callable returning a scalar reward. Invoked once per calibration
        trace as ``reward_fn(**kwargs)`` where ``kwargs`` is the trace dict
        minus the meta keys (or the explicit ``reward_kwargs_keys`` whitelist).
    traces
        Calibration set: any non-empty sequence of dict-like objects. Each
        trace MUST contain the keys required by the chosen non-conformity
        score plus all kwargs ``reward_fn`` consumes.
    alpha
        Nominal miscoverage in ``(0, 1)``. Default ``0.1`` → 90% target coverage.
    nonconformity
        Either a registered name (``"scaled_residual"`` (default),
        ``"abs_residual"``, ``"binary"``), a custom callable
        ``(trace, predicted_reward) -> float``, or a
        :class:`vlabs_calibrate.nonconformity.NonconformityScore`.
    eps
        Numerical floor for the σ denominator in scale-aware scores.
    reward_kwargs_keys
        Optional whitelist of trace keys to pass to ``reward_fn``. If ``None``
        (default), all non-meta keys are forwarded.

    Returns
    -------
    CalibratedRewardFn

    Raises
    ------
    ValueError
        If ``alpha ∉ (0, 1)``, ``len(traces) < 2``, or required trace keys
        are missing.
    TypeError
        If ``nonconformity`` is neither a string, callable, nor
        :class:`NonconformityScore`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if len(traces) < 2:
        raise ValueError(f"need at least 2 calibration traces; got {len(traces)}")

    nc_score, nc_name = _resolve_nonconformity(nonconformity)

    keys_arg: tuple[str, ...] | None = (
        tuple(reward_kwargs_keys) if reward_kwargs_keys is not None else None
    )

    scores = np.empty(len(traces), dtype=np.float64)
    for i, trace in enumerate(traces):
        _validate_trace_keys(trace, nc_score.required_trace_keys, index=i, context="calibrate")
        kwargs = _extract_reward_kwargs(trace, keys_arg)
        try:
            predicted = float(reward_fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 — re-raised with index annotation
            raise type(exc)(
                f"reward_fn raised on trace[{i}] with kwargs {sorted(kwargs)}: {exc}"
            ) from exc
        try:
            score = float(nc_score.score_fn(trace, predicted, eps))
        except Exception as exc:  # noqa: BLE001
            raise type(exc)(
                f"nonconformity {nc_name!r} raised on trace[{i}]: {exc}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"non-finite non-conformity score {score!r} at trace[{i}]; "
                f"check the {nc_name!r} score implementation"
            )
        scores[i] = score

    quantile = core.split_conformal_quantile(scores, alpha=alpha)
    nc_stats = _summarize_scores(scores)

    return CalibratedRewardFn(
        reward_fn=reward_fn,
        quantile=quantile,
        alpha=float(alpha),
        n_calibration=len(traces),
        nonconformity_name=nc_name,
        nonconformity_stats=nc_stats,
        reward_kwargs_keys=keys_arg,
        eps=float(eps),
        _nc_score=nc_score,
    )


__all__ = [
    "calibrate",
    "CalibratedRewardFn",
]
