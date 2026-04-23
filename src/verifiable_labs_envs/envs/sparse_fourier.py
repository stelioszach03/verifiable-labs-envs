"""Environment 1 — sparse Fourier recovery (1D compressed sensing).

The canonical test problem for compressed sensing and the most direct bridge from
VLBI / aperture-synthesis imaging (where the forward operator is also a subsampled
Fourier transform) to a clean, synthetic, fully verifiable RL environment.

Problem:
    y = S . F(x*) + eta,    eta ~ N(0, sigma^2 I)

where ``F`` is the orthonormal 1D DFT, ``S`` is a binary subsampling operator
selecting ``m`` of ``n`` frequencies, and ``x* in R^n`` is ``k``-sparse with
standard-normal amplitudes at ``k`` uniformly-chosen positions.

Reward is a weighted sum of:
  * NMSE term  — reconstruction quality, ``exp(-NMSE / tau)``.
  * Support-F1 term — sparsity-pattern recovery on the estimated top-k support.
  * Conformal-coverage term — reward peaks when empirical coverage of the
    standardized-conformal intervals matches the target ``1 - alpha``. This is the
    differentiator: a predictor that outputs unrealistic (too-narrow or too-wide)
    uncertainty is penalized regardless of how good its point estimate is.

Every call to ``generate_instance`` regenerates ``x*`` and ``eta`` from the seed,
so fixed-string memorization is structurally impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from verifiable_labs_envs.conformal import (
    coverage,
    coverage_score,
    interval,
    scaled_residuals,
    split_conformal_quantile,
)
from verifiable_labs_envs.forward_ops import (
    sparse_fourier_adjoint,
    sparse_fourier_forward,
    sparse_fourier_sample_mask,
)

NAME = "sparse-fourier-recovery"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "n": 256,
    "m": 64,
    "k": 10,
    "sigma": 0.05,
    "alpha": 0.1,
}

DEFAULT_WEIGHTS: dict[str, float] = {"nmse": 0.4, "support": 0.3, "conformal": 0.3}
NMSE_TAU: float = 0.5  # score_nmse = exp(-NMSE / NMSE_TAU)


@dataclass(frozen=True)
class Instance:
    """One problem draw. ``x_true`` and ``support_true`` are the oracle fields used
    by the scorer only; a solver must treat them as hidden."""

    y: np.ndarray
    mask: np.ndarray
    sigma: float
    n: int
    k: int
    alpha: float
    x_true: np.ndarray
    support_true: np.ndarray
    seed: int

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs passed to a solver (excludes the oracle fields)."""
        return {
            "y": self.y,
            "mask": self.mask,
            "sigma": self.sigma,
            "n": self.n,
            "k": self.k,
        }


@dataclass(frozen=True)
class Prediction:
    """A solver's answer.

    ``sigma_hat`` is required and must be strictly positive entry-wise — the
    conformal reward needs a non-degenerate per-entry uncertainty. Solvers that
    only produce a point estimate can pass ``sigma_hat = np.ones_like(x_hat)`` as
    a trivial default; the environment will then score a wide-and-uninformative
    interval and the conformal-coverage reward will reflect that.
    """

    x_hat: np.ndarray
    sigma_hat: np.ndarray
    support_hat: np.ndarray | None = None


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    """Sample a fresh ``k``-sparse signal, apply the forward operator, add noise."""
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    n = int(params["n"])
    m = int(params["m"])
    k = int(params["k"])
    sigma = float(params["sigma"])
    alpha = float(params["alpha"])
    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")
    if m > n:
        raise ValueError(f"m={m} must be <= n={n}")

    rng = np.random.default_rng(seed)

    support = np.sort(rng.choice(n, size=k, replace=False))
    x_true = np.zeros(n, dtype=np.float64)
    x_true[support] = rng.standard_normal(k)

    mask = sparse_fourier_sample_mask(n, m, rng)
    y_clean = sparse_fourier_forward(x_true, mask)
    noise = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) * (sigma / np.sqrt(2.0))
    y = y_clean + noise

    return Instance(
        y=y,
        mask=mask,
        sigma=sigma,
        n=n,
        k=k,
        alpha=alpha,
        x_true=x_true,
        support_true=support,
        seed=int(seed),
    )


def _omp_single(
    y: np.ndarray,
    mask: np.ndarray,
    n: int,
    k: int,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One OMP (orthogonal matching pursuit) run for k-sparse recovery.

    Known ``k``. At each step the column of ``A`` most correlated with the
    current residual is added to the active set and the real-valued
    least-squares refit is performed on the full active set. Returns the
    length-``n`` real recovery and the active support (sorted).

    OMP is used instead of ISTA/FISTA because it does not suffer the LASSO
    shrinkage bias — critical for achieving low NMSE on our default
    4x-undersampled problem where shrinkage would otherwise dominate the error.
    """
    steps = k if max_steps is None else min(int(max_steps), int(k))
    support: list[int] = []
    residual = y.copy()
    x_hat = np.zeros(n, dtype=np.float64)

    for _ in range(steps):
        corr = np.abs(sparse_fourier_adjoint(residual, mask, n))
        for s in support:
            corr[s] = -1.0
        new_idx = int(np.argmax(corr))
        support.append(new_idx)

        s_arr = np.array(sorted(support), dtype=np.int64)
        # Column j of A corresponds to frequency index s_arr[j]:
        # A[m, j] = (1/sqrt(n)) * exp(-2pi i mask[m] * s_arr[j] / n)
        A_s = np.exp(-2j * np.pi * np.outer(mask, s_arr) / n) / np.sqrt(n)
        A_stacked = np.vstack([A_s.real, A_s.imag])
        y_stacked = np.concatenate([y.real, y.imag])
        x_s, *_ = np.linalg.lstsq(A_stacked, y_stacked, rcond=None)

        x_hat = np.zeros(n, dtype=np.float64)
        x_hat[s_arr] = x_s
        residual = y - A_s @ x_s

    return x_hat, np.array(sorted(support), dtype=np.int64)


def ista_baseline(
    y: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    n: int,
    k: int,
    lam: float | None = None,  # noqa: ARG001 -- retained for API stability; unused
    n_iters: int = 200,
    n_bootstrap: int = 20,  # noqa: ARG001 -- retained for API stability; unused
    seed: int = 0,  # noqa: ARG001 -- retained for API stability; unused
) -> Prediction:
    """Reference sparse-recovery baseline using OMP with LS-covariance uncertainty.

    The internal solver is OMP (see ``_omp_single``). The public name
    ``ista_baseline`` is retained for API stability — callers treat this as
    "the reference compressed-sensing baseline." ``lam``, ``n_bootstrap``, and
    ``seed`` are accepted but unused; ``n_iters`` caps the OMP step count
    (useful when testing with ``n_iters < k``).

    Per-entry ``sigma_hat`` is built from two pieces. On the OMP-selected
    support, the closed-form least-squares standard error::

        sigma_hat_S = (sigma / sqrt(2)) * sqrt(diag( (A_S^T A_S)^-1 ))

    where ``A_S`` is the real-valued stacked forward operator on the support
    (stacking real and imaginary parts so we can run real-valued LS against a
    complex measurement). Off the selected support, ``sigma_hat`` is set to the
    **signal-amplitude prior scale** (std of the nonzero entries of ``x*``,
    which for this environment is 1.0). That value reflects the honest "I have
    no information here" uncertainty: if OMP misses a true-support entry, the
    residual at that entry is on the order of the signal amplitude, and
    matching the reported sigma_hat to the same scale keeps the standardized
    residual ~O(1) rather than blowing up when dividing by a tiny floor.
    """
    x_hat, support_hat = _omp_single(y, mask, n, k, max_steps=n_iters)

    # Off-support prior scale matches the signal-amplitude prior in
    # generate_instance (nonzero entries drawn from N(0, 1)).
    sigma_hat = np.full(n, 1.0, dtype=np.float64)
    if support_hat.size > 0:
        A_s = np.exp(-2j * np.pi * np.outer(mask, support_hat) / n) / np.sqrt(n)
        A_stacked = np.vstack([A_s.real, A_s.imag])
        try:
            cov_s = (float(sigma) ** 2 / 2.0) * np.linalg.inv(A_stacked.T @ A_stacked)
            sigma_hat_s = np.sqrt(np.maximum(np.diag(cov_s), 0.0))
        except np.linalg.LinAlgError:
            sigma_hat_s = np.full(support_hat.size, float(sigma), dtype=np.float64)
        sigma_hat[support_hat] = np.maximum(sigma_hat_s, 1e-6)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat, support_hat=support_hat)


def zero_baseline(
    y: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    n: int,
    k: int,
    **_: Any,
) -> Prediction:
    """Trivial lower-bound solver — predicts zero everywhere with unit uncertainty."""
    return Prediction(
        x_hat=np.zeros(n, dtype=np.float64),
        sigma_hat=np.ones(n, dtype=np.float64),
        support_hat=np.array([], dtype=np.int64),
    )


def _support_f1(
    support_hat: np.ndarray | None,
    support_true: np.ndarray,
    x_hat: np.ndarray,
    k: int,
) -> float:
    if support_hat is None or support_hat.size == 0:
        support_hat = np.sort(np.argpartition(np.abs(x_hat), -k)[-k:])
    sh = {int(i) for i in support_hat}
    st = {int(i) for i in support_true}
    tp = len(sh & st)
    if tp == 0:
        return 0.0
    fp = len(sh - st)
    fn = len(st - sh)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


class SparseFourierEnv:
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

    def generate_instance(self, seed: int, **kwargs: Any) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, **merged)

    def score(
        self,
        prediction: Prediction,
        instance: Instance,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        w = {**self.weights, **(weights or {})}

        nmse_raw = float(
            np.sum((prediction.x_hat - instance.x_true) ** 2)
            / max(float(np.sum(instance.x_true ** 2)), 1e-12)
        )
        score_nmse = float(np.exp(-nmse_raw / NMSE_TAU))

        score_support = _support_f1(
            prediction.support_hat, instance.support_true, prediction.x_hat, instance.k
        )

        lo, hi = interval(prediction.x_hat, prediction.sigma_hat, self.conformal_quantile)
        # Coverage is measured on the support entries only — zero entries are
        # trivial to cover (both x_true and x_hat are approximately zero there)
        # and would wash out the uncertainty-calibration signal we want to reward.
        s = instance.support_true
        cov = coverage(instance.x_true[s], lo[s], hi[s])
        target = 1.0 - instance.alpha
        score_conf = coverage_score(cov, target)

        reward = (
            w["nmse"] * score_nmse
            + w["support"] * score_support
            + w["conformal"] * score_conf
        )
        return {
            "reward": float(reward),
            "components": {
                "nmse": score_nmse,
                "support": score_support,
                "conformal": score_conf,
            },
            "meta": {
                "nmse_raw": nmse_raw,
                "coverage": cov,
                "target_coverage": target,
                "conformal_quantile": self.conformal_quantile,
                "weights": dict(w),
            },
        }

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Convenience — generate one instance, solve with ISTA, return the score dict."""
        instance = self.generate_instance(seed, **kwargs)
        prediction = ista_baseline(**instance.as_inputs(), seed=seed)
        return self.score(prediction, instance)


def calibrate_conformal_quantile(
    n_samples: int = 500,
    alpha: float = DEFAULT_HYPERPARAMS["alpha"],
    n_bootstrap: int = 20,
    n_iters: int = 200,
    hyperparams: dict[str, Any] | None = None,
    start_seed: int = 0,
) -> float:
    """Run the reference baseline on ``n_samples`` fresh instances, pool the
    per-entry standardized residuals ``|x_hat - x*| / sigma_hat`` **at the
    support entries only**, and return the ``(1 - alpha)`` split-conformal
    quantile.

    Support-only calibration is essential: the 246-of-256 zero entries in a
    typical 10-sparse 256-dim instance have near-zero residuals regardless of
    solver quality, and would collapse the pooled quantile to 0 and with it
    the reward signal we want to produce.
    """
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    residuals_list = []
    for i in range(n_samples):
        inst = generate_instance(seed=start_seed + i, **params)
        pred = ista_baseline(
            **inst.as_inputs(),
            n_iters=n_iters,
            n_bootstrap=n_bootstrap,
            seed=start_seed + i,
        )
        r = scaled_residuals(pred.x_hat, inst.x_true, pred.sigma_hat)
        residuals_list.append(r[inst.support_true])
    pooled = np.concatenate(residuals_list)
    return split_conformal_quantile(pooled, alpha)


@lru_cache(maxsize=8)
def _cached_quantile(
    n_samples: int,
    alpha: float,
    n_bootstrap: int,
    n_iters: int,
) -> float:
    return calibrate_conformal_quantile(
        n_samples=n_samples,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_iters=n_iters,
    )


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> SparseFourierEnv:
    """Factory. If ``calibration_quantile`` is provided, skip calibration.

    With ``fast=True`` (default) the first call runs a 30-sample, 5-bootstrap,
    80-iter ISTA calibration (~few seconds) and caches the result. With
    ``fast=False`` the call runs a 500-sample, 20-bootstrap, 200-iter calibration
    (~30 s) matching the values baked into the environment design. Pass
    ``calibration_quantile`` explicitly in tests to skip calibration entirely.
    """
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(
            n_samples=30,
            alpha=float(DEFAULT_HYPERPARAMS["alpha"]),
            n_bootstrap=5,
            n_iters=80,
        )
    else:
        q = _cached_quantile(
            n_samples=500,
            alpha=float(DEFAULT_HYPERPARAMS["alpha"]),
            n_bootstrap=20,
            n_iters=200,
        )
    return SparseFourierEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "Instance",
    "Prediction",
    "SparseFourierEnv",
    "generate_instance",
    "ista_baseline",
    "zero_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
]
