"""Tests for the public dataclasses in vlabs_calibrate.types."""
from __future__ import annotations

import dataclasses

import pytest

from vlabs_calibrate.types import CalibrationResult, CoverageReport


def test_calibration_result_is_frozen() -> None:
    r = CalibrationResult(
        reward=0.5,
        interval=(0.1, 0.9),
        sigma=0.2,
        quantile=2.0,
        alpha=0.1,
        target_coverage=0.9,
        covered=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.reward = 0.6  # type: ignore[misc]


def test_calibration_result_default_covered_is_none() -> None:
    r = CalibrationResult(
        reward=0.5,
        interval=(0.1, 0.9),
        sigma=0.2,
        quantile=2.0,
        alpha=0.1,
        target_coverage=0.9,
    )
    assert r.covered is None


def test_coverage_report_is_frozen() -> None:
    rep = CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.91,
        n=100,
        n_in_interval=91,
        interval_width_mean=1.2,
        interval_width_median=1.1,
        nonconformity={"mean": 0.5},
        quantile=2.0,
        alpha=0.1,
        tolerance=0.05,
        passes=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.passes = False  # type: ignore[misc]


def test_coverage_report_passes_is_explicit() -> None:
    """``passes`` is set by the caller, not derived; CoverageReport stays inert."""
    # Caller can produce a "failing" report by setting passes=False even when
    # |empirical - target| <= tolerance — the dataclass does not second-guess.
    rep = CoverageReport(
        target_coverage=0.9,
        empirical_coverage=0.9,  # equal to target
        n=10,
        n_in_interval=9,
        interval_width_mean=1.0,
        interval_width_median=1.0,
        nonconformity={},
        quantile=1.0,
        alpha=0.1,
        tolerance=0.05,
        passes=False,
    )
    assert rep.passes is False
