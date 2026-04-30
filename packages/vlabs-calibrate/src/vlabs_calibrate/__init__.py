"""vlabs-calibrate — conformal coverage guarantees for any reward function.

Public entry point::

    import vlabs_calibrate as vc

    calibrated = vc.calibrate(my_reward_fn, traces, alpha=0.1)
    result = calibrated(prompt=..., completion=..., ground_truth=..., sigma=0.5)
    print(result.reward, result.interval, result.target_coverage)

The package wraps any Python reward callable with a split-conformal
prediction interval providing marginal ``(1 − α)`` coverage under
exchangeability.

See :mod:`vlabs_calibrate.calibrate` for the public API and
:mod:`vlabs_calibrate.core` for the underlying primitives.
"""
from __future__ import annotations

from vlabs_calibrate import core, nonconformity
from vlabs_calibrate._version import __version__
from vlabs_calibrate.calibrate import CalibratedRewardFn, calibrate
from vlabs_calibrate.types import CalibrationResult, CoverageReport, Trace

__all__ = [
    "__version__",
    "calibrate",
    "CalibratedRewardFn",
    "CalibrationResult",
    "CoverageReport",
    "Trace",
    "core",
    "nonconformity",
]
