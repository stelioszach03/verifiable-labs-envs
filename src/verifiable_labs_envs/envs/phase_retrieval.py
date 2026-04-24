"""Environment — phase retrieval (1D compressed sensing with magnitude-only measurement).

Canonical in X-ray crystallography, coherent diffraction imaging, semiconductor
metrology, and astronomical speckle interferometry: recover ``x`` from
``y = |F(x)|`` — the magnitude of its Fourier transform, with phase discarded.
The classical workhorse is Gerchberg-Saxton iteration.

Problem:
    y = |S · F(x*)| + eta,    eta ~ N(0, sigma^2 I) (on the magnitude)

where ``F`` is the orthonormal 1D DFT, ``S`` selects ``m`` of ``n`` frequency
positions, and ``x* ∈ R^n`` is ``k``-sparse with standard-normal amplitudes.

Sign ambiguity: because ``|F(-x)| = |F(x)|``, recovery is only unique up to a
global sign flip. The scorer takes the min NMSE between ``x_hat`` and both
``+x_true`` and ``-x_true``.

Reward components:
  * NMSE (sign-invariant) — ``exp(-NMSE / tau)``.
  * Support-F1 — top-k support recovery (sign-invariant).
  * Conformal coverage — same split-conformal calibration as sparse-fourier,
    support-only to avoid the zero-entry dominance.
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
    MagnitudeOnlyOp,
    sparse_fourier_sample_mask,
)

NAME = "phase-retrieval"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "n": 32,
    "m": 24,
    "k": 4,
    "sigma": 0.02,  # magnitude-domain noise (smaller than sparse-F because mags are smaller)
    "alpha": 0.1,
    "gs_iters": 200,
}

DEFAULT_WEIGHTS: dict[str, float] = {"nmse": 0.4, "support": 0.3, "conformal": 0.3}
NMSE_TAU: float = 0.5


@dataclass(frozen=True)
class Instance:
    """Phase-retrieval problem instance. ``y`` is the measured magnitude."""

    y: np.ndarray  # real non-negative, shape (m,)
    mask: np.ndarray  # int, shape (m,)
    sigma: float
    n: int
    k: int
    alpha: float
    x_true: np.ndarray  # real, shape (n,)
    support_true: np.ndarray
    seed: int

    def as_inputs(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "mask": self.mask,
            "sigma": self.sigma,
            "n": self.n,
            "k": self.k,
        }


@dataclass(frozen=True)
class Prediction:
    x_hat: np.ndarray
    sigma_hat: np.ndarray
    support_hat: np.ndarray | None = None


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    """Sample a fresh k-sparse real signal, apply |F|, add magnitude-domain noise."""
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
    op = MagnitudeOnlyOp(n=n, mask=mask)
    y_clean = op.apply(x_true)  # real non-negative, shape (m,)
    noise = sigma * rng.standard_normal(m)
    y = np.maximum(y_clean + noise, 0.0)  # clip to non-negative (magnitudes can't be < 0)

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


def _project_k_sparse(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (projected, support): zero out all but top-k entries by |value|."""
    if k >= x.size:
        return x.copy(), np.arange(x.size, dtype=np.int64)
    idx = np.argpartition(np.abs(x), -k)[-k:]
    idx.sort()
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out, idx.astype(np.int64)


def gerchberg_saxton_baseline(
    y: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    n: int,
    k: int,
    gs_iters: int = 200,
    n_restarts: int = 5,
    seed: int = 0,
) -> Prediction:
    """Classical Gerchberg-Saxton with k-sparse projection and random restarts.

    Alternates between:
      - forward: compute F(x), replace magnitudes with measured y (at mask) and
        keep estimated phases.
      - backward: inverse DFT, project onto k-sparse real support.

    Random restarts pick different initial phases; the best NMSE among
    ``n_restarts`` candidates is returned (by |F(x̂)| residual vs y).
    """
    best_x_hat: np.ndarray | None = None
    best_support: np.ndarray | None = None
    best_residual = np.inf
    rng = np.random.default_rng(seed + 10_000)

    for _restart in range(n_restarts):
        # Initialize phases
        phi = rng.uniform(0, 2 * np.pi, size=mask.size)
        # Coefficients on the mask: magnitude = y, phase = phi
        Z = np.zeros(n, dtype=np.complex128)
        Z[mask] = y * np.exp(1j * phi)
        x_hat = np.real(np.fft.ifft(Z, norm="ortho"))

        for _ in range(gs_iters):
            # Project onto k-sparse real
            x_hat, _ = _project_k_sparse(x_hat, k)
            # Forward
            X = np.fft.fft(x_hat, norm="ortho")
            # Replace magnitudes on the mask with measured y, keep phase
            mag = np.abs(X[mask])
            mag_safe = np.where(mag > 1e-12, mag, 1.0)
            new_coeffs = y * X[mask] / mag_safe
            Z = np.zeros(n, dtype=np.complex128)
            Z[mask] = new_coeffs
            # Inverse DFT, take real part
            x_hat = np.real(np.fft.ifft(Z, norm="ortho"))

        # Final projection + support
        x_hat, support_hat = _project_k_sparse(x_hat, k)

        # Compute measurement residual to pick best restart
        X_final = np.fft.fft(x_hat, norm="ortho")
        residual_magnitude = float(np.linalg.norm(np.abs(X_final[mask]) - y))
        if residual_magnitude < best_residual:
            best_residual = residual_magnitude
            best_x_hat = x_hat
            best_support = support_hat

    assert best_x_hat is not None and best_support is not None

    # sigma_hat: flat at signal-amplitude prior (1.0) off-support, smaller on-support.
    sigma_hat = np.full(n, 1.0, dtype=np.float64)
    sigma_hat[best_support] = max(float(sigma) * 4.0, 0.1)  # rough on-support sigma

    return Prediction(
        x_hat=best_x_hat,
        sigma_hat=sigma_hat,
        support_hat=best_support,
    )


def zero_baseline(
    y: np.ndarray,  # noqa: ARG001
    mask: np.ndarray,  # noqa: ARG001
    sigma: float,  # noqa: ARG001
    n: int,
    k: int,  # noqa: ARG001
    **_: Any,
) -> Prediction:
    return Prediction(
        x_hat=np.zeros(n, dtype=np.float64),
        sigma_hat=np.ones(n, dtype=np.float64),
        support_hat=np.array([], dtype=np.int64),
    )


def _sign_invariant_nmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Min NMSE between x_hat and {+x_true, -x_true}. Accounts for the
    |F(x)| = |F(-x)| global-sign ambiguity."""
    denom = max(float(np.sum(x_true ** 2)), 1e-12)
    nmse_plus = float(np.sum((x_true - x_hat) ** 2)) / denom
    nmse_minus = float(np.sum((x_true + x_hat) ** 2)) / denom
    return min(nmse_plus, nmse_minus)


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


class PhaseRetrievalEnv:
    """Phase-retrieval env — same API shape as SparseFourierEnv."""

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

        nmse_raw = _sign_invariant_nmse(instance.x_true, prediction.x_hat)
        score_nmse = float(np.exp(-nmse_raw / NMSE_TAU))

        # Sign-invariant support F1: use |x_hat| for support estimation.
        score_support = _support_f1(
            prediction.support_hat, instance.support_true, prediction.x_hat, instance.k
        )

        # Conformal interval on the aligned (sign-flipped) prediction.
        # Pick the sign that minimizes NMSE, then compute coverage on that.
        plus = float(np.sum((instance.x_true - prediction.x_hat) ** 2))
        minus = float(np.sum((instance.x_true + prediction.x_hat) ** 2))
        aligned_x_hat = prediction.x_hat if plus <= minus else -prediction.x_hat

        lo, hi = interval(aligned_x_hat, prediction.sigma_hat, self.conformal_quantile)
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
                "sign_aligned_to": "plus" if plus <= minus else "minus",
            },
        }

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        instance = self.generate_instance(seed, **kwargs)
        prediction = gerchberg_saxton_baseline(
            **instance.as_inputs(),
            gs_iters=int(self.hyperparams.get("gs_iters", 200)),
            seed=seed,
        )
        return self.score(prediction, instance)


def calibrate_conformal_quantile(
    n_samples: int = 200,
    alpha: float = DEFAULT_HYPERPARAMS["alpha"],
    gs_iters: int = 100,
    hyperparams: dict[str, Any] | None = None,
    start_seed: int = 10_000,
) -> float:
    """Support-only split-conformal calibration on Gerchberg-Saxton residuals."""
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    residuals_list = []
    for i in range(n_samples):
        inst = generate_instance(seed=start_seed + i, **params)
        pred = gerchberg_saxton_baseline(**inst.as_inputs(), gs_iters=gs_iters, seed=start_seed + i)
        # Align sign before residual
        plus = np.sum((inst.x_true - pred.x_hat) ** 2)
        minus = np.sum((inst.x_true + pred.x_hat) ** 2)
        x_hat_aligned = pred.x_hat if plus <= minus else -pred.x_hat
        r = scaled_residuals(x_hat_aligned, inst.x_true, pred.sigma_hat)
        residuals_list.append(r[inst.support_true])
    pooled = np.concatenate(residuals_list)
    return split_conformal_quantile(pooled, alpha)


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float, gs_iters: int) -> float:
    return calibrate_conformal_quantile(
        n_samples=n_samples, alpha=alpha, gs_iters=gs_iters
    )


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> PhaseRetrievalEnv:
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(
            n_samples=30, alpha=float(DEFAULT_HYPERPARAMS["alpha"]), gs_iters=50,
        )
    else:
        q = _cached_quantile(
            n_samples=200, alpha=float(DEFAULT_HYPERPARAMS["alpha"]), gs_iters=200,
        )
    return PhaseRetrievalEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "Instance",
    "Prediction",
    "PhaseRetrievalEnv",
    "generate_instance",
    "gerchberg_saxton_baseline",
    "zero_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
]
