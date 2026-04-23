"""Environment 3 — simplified low-dose 2D parallel-beam CT reconstruction.

    y = R(x*) + eta,    eta ~ N(0, sigma^2 I)

where ``R`` is the 2D parallel-beam Radon transform at ``n_angles`` angles (4x
undersampled relative to the full-angular Nyquist rate for this image size).
``x*`` is a grayscale phantom in ``[0, 1]`` serving as a stand-in for linear
attenuation. Full LoDoPaB-CT integration (Poisson photon-counting model, real
clinical phantoms) is a follow-up; this environment ships the Shepp-Logan
phantom plus a rotation through natural-image phantoms as the v0.0.1 source.

Reward is a weighted sum of:
  * PSNR — mapped linearly from 15 dB -> 0 to 35 dB -> 1.
  * SSIM — ``skimage.metrics.structural_similarity`` clipped to ``[0, 1]``.
  * Conformal coverage — pooled per-pixel standardized residuals, calibrated
    on the FBP baseline.

Each call to ``generate_instance`` regenerates the sinogram noise from the
seed, keeping the environment contamination-resistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from skimage import data as skdata
from skimage import transform as sktransform
from skimage.metrics import structural_similarity as ssim_metric

from verifiable_labs_envs.conformal import (
    coverage,
    coverage_score,
    interval,
    scaled_residuals,
    split_conformal_quantile,
)
from verifiable_labs_envs.forward_ops import radon_fbp, radon_forward

NAME = "lodopab-ct-simplified"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "shape": (128, 128),
    "n_angles": 60,       # 4x undersampled from ~240 full-angular at 128-px side
    "noise_sigma": 0.5,   # relative to sinogram amplitude ~10-30
    "alpha": 0.1,
}

DEFAULT_WEIGHTS: dict[str, float] = {"psnr": 0.4, "ssim": 0.3, "conformal": 0.3}

CALIBRATION_PHANTOMS: tuple[str, ...] = (
    "shepp_logan",
    "moon",
    "camera",
    "astronaut",
    "coffee",
)

PSNR_MIN_DB: float = 15.0
PSNR_MAX_DB: float = 35.0


@dataclass(frozen=True)
class Instance:
    y: np.ndarray  # sinogram, shape (n_side, n_angles)
    x_true: np.ndarray  # phantom, shape (n_side, n_side)
    angles_deg: np.ndarray
    shape: tuple[int, int]
    n_angles: int
    noise_sigma: float
    alpha: float
    phantom_name: str
    seed: int

    def as_inputs(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "angles_deg": self.angles_deg,
            "shape": self.shape,
            "n_angles": self.n_angles,
            "noise_sigma": self.noise_sigma,
        }


@dataclass(frozen=True)
class Prediction:
    x_hat: np.ndarray
    sigma_hat: np.ndarray


@lru_cache(maxsize=16)
def _load_phantom(phantom_name: str, shape: tuple[int, int]) -> np.ndarray:
    """Load a 2D grayscale phantom, resize to ``shape``, and normalize to [0, 1]."""
    if phantom_name == "shepp_logan":
        if not hasattr(skdata, "shepp_logan_phantom"):
            raise KeyError("skimage.data has no shepp_logan_phantom in this version")
        img = skdata.shepp_logan_phantom().astype(np.float64)
    elif hasattr(skdata, phantom_name):
        img = getattr(skdata, phantom_name)()
        img = (
            img.astype(np.float64).mean(axis=-1)
            if img.ndim == 3
            else img.astype(np.float64)
        )
    else:
        raise KeyError(f"Unknown phantom '{phantom_name}'")
    img = sktransform.resize(img, shape, anti_aliasing=True, mode="reflect")
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    # Apply a circular mask so the phantom is zero outside the inscribed disc.
    # parallel-beam Radon with ``circle=True`` requires this; also suppresses
    # the corresponding skimage warning.
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 1.0
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img = img * mask
    return img.astype(np.float64)


def _angles_for(n_angles: int) -> np.ndarray:
    """Equispaced projection angles in [0, 180) degrees."""
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def generate_instance(
    seed: int,
    phantom_name: str | None = None,
    **kwargs: Any,
) -> Instance:
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    shape = tuple(params["shape"])
    n_angles = int(params["n_angles"])
    noise_sigma = float(params["noise_sigma"])
    alpha = float(params["alpha"])

    if phantom_name is None:
        phantom_name = CALIBRATION_PHANTOMS[seed % len(CALIBRATION_PHANTOMS)]

    x_true = _load_phantom(phantom_name, shape)
    angles = _angles_for(n_angles)
    y_clean = radon_forward(x_true, angles)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(y_clean.shape) * noise_sigma
    y = y_clean + noise

    return Instance(
        y=y,
        x_true=x_true,
        angles_deg=angles,
        shape=shape,
        n_angles=n_angles,
        noise_sigma=noise_sigma,
        alpha=alpha,
        phantom_name=phantom_name,
        seed=int(seed),
    )


def fbp_baseline(
    y: np.ndarray,
    angles_deg: np.ndarray,
    shape: tuple[int, int],
    n_angles: int,  # noqa: ARG001 -- inferred from angles_deg
    noise_sigma: float,
    **_: Any,
) -> Prediction:
    """Filtered back-projection with a spatially-varying sigma_hat.

    The point estimate is classical Ram-Lak FBP. ``sigma_hat`` blends a noise
    floor and a Sobel-gradient term so that edge pixels (where FBP streak
    artifacts concentrate) receive honestly larger uncertainty.
    """
    x_hat = radon_fbp(y, angles_deg, output_size=shape[0]).astype(np.float64)

    from scipy.ndimage import sobel

    gx = sobel(x_hat, axis=0, mode="reflect")
    gy = sobel(x_hat, axis=1, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad / (float(grad.max()) + 1e-8)
    sigma_hat = 0.02 + 2.0 * float(noise_sigma) * (0.2 + grad_norm)
    sigma_hat = np.maximum(sigma_hat, 1e-4)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


def zero_baseline(
    y: np.ndarray,
    shape: tuple[int, int],
    **_: Any,
) -> Prediction:
    """Trivial constant-image lower-bound baseline."""
    mean_val = float(y.mean()) / max(float(y.max()), 1.0)
    x_hat = np.full(shape, np.clip(mean_val, 0.0, 1.0), dtype=np.float64)
    sigma_hat = np.full(shape, 0.5, dtype=np.float64)
    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


def _score_psnr(x_hat: np.ndarray, x_true: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((x_hat - x_true) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    score = float(np.clip((psnr - PSNR_MIN_DB) / (PSNR_MAX_DB - PSNR_MIN_DB), 0.0, 1.0))
    return score, psnr


def _score_ssim(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    value = ssim_metric(x_true, x_hat, data_range=1.0)
    return float(max(0.0, min(1.0, value)))


class LodopabCtEnv:
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

    def generate_instance(
        self, seed: int, phantom_name: str | None = None, **kwargs: Any
    ) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, phantom_name=phantom_name, **merged)

    def score(
        self,
        prediction: Prediction,
        instance: Instance,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        w = {**self.weights, **(weights or {})}

        score_psnr, psnr_db = _score_psnr(prediction.x_hat, instance.x_true)
        score_ssim = _score_ssim(prediction.x_hat, instance.x_true)

        lo, hi = interval(prediction.x_hat, prediction.sigma_hat, self.conformal_quantile)
        cov = coverage(instance.x_true, lo, hi)
        target = 1.0 - instance.alpha
        score_conf = coverage_score(cov, target)

        reward = (
            w["psnr"] * score_psnr
            + w["ssim"] * score_ssim
            + w["conformal"] * score_conf
        )
        return {
            "reward": float(reward),
            "components": {
                "psnr": score_psnr,
                "ssim": score_ssim,
                "conformal": score_conf,
            },
            "meta": {
                "psnr_db": psnr_db,
                "coverage": cov,
                "target_coverage": target,
                "conformal_quantile": self.conformal_quantile,
                "weights": dict(w),
                "phantom_name": instance.phantom_name,
            },
        }

    def run_baseline(
        self, seed: int = 0, phantom_name: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        instance = self.generate_instance(seed, phantom_name=phantom_name, **kwargs)
        prediction = fbp_baseline(**instance.as_inputs())
        return self.score(prediction, instance)


def calibrate_conformal_quantile(
    alpha: float = DEFAULT_HYPERPARAMS["alpha"],
    n_samples: int | None = None,
    hyperparams: dict[str, Any] | None = None,
    phantoms: tuple[str, ...] = CALIBRATION_PHANTOMS,
) -> float:
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    names = phantoms[:n_samples] if n_samples is not None else phantoms
    residuals_list = []
    for seed, name in enumerate(names):
        inst = generate_instance(seed=seed, phantom_name=name, **params)
        pred = fbp_baseline(**inst.as_inputs())
        r = scaled_residuals(pred.x_hat, inst.x_true, pred.sigma_hat)
        residuals_list.append(r.flatten())
    pooled = np.concatenate(residuals_list)
    return split_conformal_quantile(pooled, alpha)


@lru_cache(maxsize=8)
def _cached_quantile(alpha: float, n_samples: int | None) -> float:
    return calibrate_conformal_quantile(alpha=alpha, n_samples=n_samples)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> LodopabCtEnv:
    """Factory. Fast calibration uses the first three phantoms; full uses all five."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), 3)
    else:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), None)
    return LodopabCtEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "CALIBRATION_PHANTOMS",
    "Instance",
    "Prediction",
    "LodopabCtEnv",
    "generate_instance",
    "fbp_baseline",
    "zero_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
]
