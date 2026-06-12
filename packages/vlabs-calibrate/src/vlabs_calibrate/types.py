"""Public dataclasses and the ``Trace`` TypedDict.

These types are part of the public surface (re-exported from the package
root) and are stable for ``0.1.0a1``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict


class Trace(TypedDict, total=False):
    """Loose dict shape for calibration traces.

    The required keys depend on the chosen non-conformity score:

    * ``"scaled_residual"`` (default): ``"reference_reward"`` + ``"uncertainty"``,
      plus all kwargs the user's ``reward_fn`` consumes.
    * ``"abs_residual"`` and ``"binary"``: ``"reference_reward"`` + ``reward_fn`` kwargs.

    The reserved key ``"_meta"`` is never forwarded to the reward function;
    use it to carry opaque user metadata.
    """

    reference_reward: float
    uncertainty: float
    _meta: dict[str, Any]


@dataclass(frozen=True)
class CalibrationResult:
    """Single-call output of a :class:`CalibratedRewardFn`.

    Attributes
    ----------
    reward
        The raw scalar returned by the user's ``reward_fn``.
    interval
        ``(lower, upper)`` conformal interval. For scale-aware non-conformity
        scores the interval is ``[reward − q * σ, reward + q * σ]``; for
        scale-free scores ``σ`` is ignored.
    sigma
        The σ used to construct the interval (``0.0`` for scale-free scores).
    quantile
        The calibrated ``(1 − α)`` non-conformity quantile from the parent
        :class:`CalibratedRewardFn`.
    alpha
        Nominal miscoverage level used at calibration time.
    target_coverage
        ``1 − alpha``.
    covered
        ``True`` / ``False`` if the caller passed a ``reference=`` reward and
        it falls inside ``interval``; ``None`` otherwise.
    """

    reward: float
    interval: tuple[float, float]
    sigma: float
    quantile: float
    alpha: float
    target_coverage: float
    covered: bool | None = None


@dataclass(frozen=True)
class CoverageReport:
    """Aggregate diagnostics from :meth:`CalibratedRewardFn.evaluate`.

    Attributes
    ----------
    target_coverage
        ``1 − alpha`` — the coverage we aim for under exchangeability.
    empirical_coverage
        Fraction of held-out traces whose ``reference_reward`` fell inside
        the predicted interval.
    n
        Total number of traces evaluated.
    n_in_interval
        Number of traces with ``reference_reward`` covered.
    interval_width_mean, interval_width_median
        Mean / median width of the predicted interval over the held-out set.
    nonconformity
        Summary stats of the non-conformity scores observed during
        evaluation: ``mean``, ``std``, ``min``, ``max``, ``median``.
    quantile, alpha
        Passthrough from the parent :class:`CalibratedRewardFn`.
    tolerance
        The slack (in absolute coverage units) used to set ``passes``.
    passes
        ``True`` iff ``|empirical_coverage − target_coverage| ≤ tolerance``.
    """

    target_coverage: float
    empirical_coverage: float
    n: int
    n_in_interval: int
    interval_width_mean: float
    interval_width_median: float
    nonconformity: Mapping[str, float]
    quantile: float
    alpha: float
    tolerance: float
    passes: bool


__all__ = [
    "Trace",
    "CalibrationResult",
    "CoverageReport",
]
