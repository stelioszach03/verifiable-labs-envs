"""Coverage validation — proves the public ``calibrate()`` wrapper hits
the nominal ``(1 − α)`` coverage on five challenging synthetic distributions.

The seeds are fixed so the empirical coverages are reproducible and can be
quoted verbatim in the README. Each parametrised case calibrates on
``n_train = 500`` traces (``alpha = 0.1`` ⇒ target coverage 0.9), then
evaluates on a held-out ``n_test = 2000`` set and asserts
``|empirical − 0.9| ≤ tolerance``.

Tolerance per distribution
--------------------------
* ``gaussian``, ``heavy_tail_t3``, ``bimodal``, ``sparse``: ``5pp``.
* ``structured_misspecified``: ``7pp`` — split-conformal still gives
  marginal coverage when σ is mis-specified, but finite-sample variance
  tightens slightly more slowly than the well-specified cases. The 7pp
  band documents this honestly rather than hiding it.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import vlabs_calibrate as vc

N_TRAIN = 500
N_TEST = 2000
ALPHA = 0.1
TARGET = 1.0 - ALPHA


def _identity_reward(*, x: float) -> float:
    return float(x)


def _gen_gaussian(n: int, seed: int, sigma: float = 0.5) -> list[dict]:
    """Well-specified Gaussian: ``ref = x + σ · N(0, 1)``.

    The reward function is the identity ``x``; ``reference_reward`` is the
    "gold" label and σ is the per-trace uncertainty (correctly reported).
    """
    rng = np.random.default_rng(seed)
    return [
        {
            "x": (x := float(rng.standard_normal())),
            "reference_reward": x + sigma * float(rng.standard_normal()),
            "uncertainty": sigma,
        }
        for _ in range(n)
    ]


def _gen_heavy_tail(n: int, seed: int, sigma: float = 0.5, df: int = 3) -> list[dict]:
    """Heavy-tailed: ``ref = x + σ · t(df=3)``. Tests robustness to outliers."""
    rng = np.random.default_rng(seed)
    return [
        {
            "x": (x := float(rng.standard_normal())),
            "reference_reward": x + sigma * float(rng.standard_t(df)),
            "uncertainty": sigma,
        }
        for _ in range(n)
    ]


def _gen_bimodal(n: int, seed: int, sigma: float = 0.5) -> list[dict]:
    """Equal-weight ±2σ bimodal. Sigma is "right" but the noise is non-Gaussian."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = float(rng.standard_normal())
        sign = float(rng.choice([-1.0, 1.0]))
        ref = x + sign * 2.0 * sigma + 0.3 * sigma * float(rng.standard_normal())
        out.append({"x": x, "reference_reward": ref, "uncertainty": sigma})
    return out


def _gen_sparse(n: int, seed: int, sigma: float = 0.5) -> list[dict]:
    """Sparse outliers: 70% zero residual, 30% large 5σ Gaussian residual."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = float(rng.standard_normal())
        ref = x if rng.random() < 0.7 else x + 5.0 * sigma * float(rng.standard_normal())
        out.append({"x": x, "reference_reward": ref, "uncertainty": sigma})
    return out


def _gen_structured(n: int, seed: int, sigma_reported: float = 0.5) -> list[dict]:
    """Mis-specified σ: true σ is 2× the reported σ on half the traces.

    The reward function never sees the true σ. Calibration uses the
    reported σ, which means the standardised residuals are inflated on
    the doubled half. Marginal split-conformal coverage still holds under
    exchangeability — this case stresses the finite-sample variance.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = float(rng.standard_normal())
        true_sigma = sigma_reported * (2.0 if rng.random() < 0.5 else 1.0)
        ref = x + true_sigma * float(rng.standard_normal())
        out.append({"x": x, "reference_reward": ref, "uncertainty": sigma_reported})
    return out


_CASES = [
    ("gaussian", _gen_gaussian, 0.05, 100, 200),
    ("heavy_tail_t3", _gen_heavy_tail, 0.05, 101, 201),
    ("bimodal", _gen_bimodal, 0.05, 102, 202),
    ("sparse", _gen_sparse, 0.05, 103, 203),
    ("structured_misspecified", _gen_structured, 0.07, 104, 204),
]


@pytest.mark.parametrize(
    "name,generator,tolerance,seed_train,seed_test",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_coverage_within_tolerance(
    name: str,
    generator: Callable[[int, int], list[dict]],
    tolerance: float,
    seed_train: int,
    seed_test: int,
) -> None:
    train = generator(N_TRAIN, seed_train)
    test = generator(N_TEST, seed_test)
    cal = vc.calibrate(_identity_reward, train, alpha=ALPHA)
    report = cal.evaluate(test, tolerance=tolerance)

    assert report.target_coverage == pytest.approx(TARGET)
    assert report.n == N_TEST
    margin = report.empirical_coverage - TARGET
    assert abs(margin) <= tolerance, (
        f"{name}: empirical={report.empirical_coverage:.4f}, target={TARGET}, "
        f"margin={margin:+.4f}, tolerance={tolerance}, quantile={report.quantile:.4f}"
    )

    # Stable, machine-readable line so the README results table can be
    # regenerated by running ``pytest -v -s -p no:cacheprovider``.
    print(
        f"\n[VALIDATION] case={name:25s} "
        f"q={report.quantile:7.4f} "
        f"emp={report.empirical_coverage:.4f} "
        f"width_med={report.interval_width_median:.4f} "
        f"width_mean={report.interval_width_mean:.4f} "
        f"tol={tolerance:.2f} "
        f"pass={report.passes}"
    )


def test_validation_suite_summary() -> None:
    """Aggregate run that powers the README results table.

    Re-runs all five distributions in one shot and prints a markdown table.
    """
    rows = []
    for name, generator, tolerance, seed_train, seed_test in _CASES:
        train = generator(N_TRAIN, seed_train)
        test = generator(N_TEST, seed_test)
        cal = vc.calibrate(_identity_reward, train, alpha=ALPHA)
        report = cal.evaluate(test, tolerance=tolerance)
        rows.append(
            {
                "name": name,
                "quantile": report.quantile,
                "empirical": report.empirical_coverage,
                "width_median": report.interval_width_median,
                "tolerance": tolerance,
                "passes": report.passes,
            }
        )

    print("\n[VALIDATION TABLE]")
    print("| distribution | n_train | n_test | target | empirical | quantile | width (median) | tol | pass |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|:-:|")
    for r in rows:
        print(
            f"| {r['name']} | {N_TRAIN} | {N_TEST} | {TARGET:.2f} | "
            f"{r['empirical']:.4f} | {r['quantile']:.4f} | {r['width_median']:.4f} | "
            f"±{r['tolerance']:.2f} | {'✅' if r['passes'] else '❌'} |"
        )

    assert all(r["passes"] for r in rows), "at least one distribution failed coverage"
