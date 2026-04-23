"""Environment 2 — 4x single-image super-resolution.

    y = D . G(x*) + eta,    eta ~ N(0, sigma^2 I)

where ``G`` is a 2D Gaussian blur and ``D`` is a stride-4 decimation. The
ground-truth ``x*`` is a grayscale natural image in ``[0, 1]``. The environment
ships a fixed rotation of public-domain ``skimage.data`` images as the default
source so that tests and calibration are self-contained; a production-grade
variant fed by DIV2K validation will follow.

Reward is a weighted sum of:
  * PSNR — mapped linearly from 15 dB -> 0 to 35 dB -> 1.
  * SSIM — ``skimage.metrics.structural_similarity`` clipped to ``[0, 1]``.
  * Conformal coverage — pooled per-pixel standardized residuals calibrate the
    ``(1 - alpha)``-quantile ``q_alpha``; at score time we report the fraction
    of pixels with ``x* in [x_hat - q * sigma_hat, x_hat + q * sigma_hat]``
    and reward the match to the target ``1 - alpha``.

Every call to ``generate_instance`` draws fresh noise with the seed so fixed
measurement byte-strings cannot be memorized.
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
from verifiable_labs_envs.forward_ops import blur_downsample

NAME = "super-resolution-div2k-x4"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "shape": (128, 128),
    "blur_sigma": 1.0,
    "factor": 4,
    "noise_sigma": 0.01,
    "alpha": 0.1,
}

DEFAULT_WEIGHTS: dict[str, float] = {"psnr": 0.4, "ssim": 0.3, "conformal": 0.3}

# Rotating set of public-domain skimage test images used as ground-truth sources.
# All are loaded, converted to grayscale, resized to the instance shape, and
# normalized to [0, 1]. This is the "training/calibration" image set for v0.0.1;
# DIV2K integration is a follow-up.
CALIBRATION_IMAGES: tuple[str, ...] = (
    "camera",
    "moon",
    "astronaut",
    "coffee",
    "chelsea",
    "immunohistochemistry",
)

# PSNR -> [0, 1] mapping endpoints (dB)
PSNR_MIN_DB: float = 15.0
PSNR_MAX_DB: float = 35.0


@dataclass(frozen=True)
class Instance:
    y: np.ndarray  # low-resolution measurement, (H // f, W // f)
    x_true: np.ndarray  # ground truth high-res image, (H, W)
    shape: tuple[int, int]
    blur_sigma: float
    factor: int
    noise_sigma: float
    alpha: float
    image_name: str
    seed: int

    def as_inputs(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "shape": self.shape,
            "blur_sigma": self.blur_sigma,
            "factor": self.factor,
            "noise_sigma": self.noise_sigma,
        }


@dataclass(frozen=True)
class Prediction:
    x_hat: np.ndarray  # HR estimate, (H, W), real
    sigma_hat: np.ndarray  # per-pixel std estimate, (H, W), positive


@lru_cache(maxsize=32)
def _load_gray(image_name: str, shape: tuple[int, int]) -> np.ndarray:
    """Load ``image_name`` from ``skimage.data``, grayscale + resize to ``shape``, normalize to [0, 1]."""
    if not hasattr(skdata, image_name):
        raise KeyError(f"skimage.data has no image '{image_name}'")
    img = getattr(skdata, image_name)()
    img = img.astype(np.float64).mean(axis=-1) if img.ndim == 3 else img.astype(np.float64)
    img = sktransform.resize(img, shape, anti_aliasing=True, mode="reflect")
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return img.astype(np.float64)


def generate_instance(seed: int, image_name: str | None = None, **kwargs: Any) -> Instance:
    """Sample one HR image, apply blur+decimate, add noise.

    If ``image_name`` is omitted the seed picks deterministically from
    ``CALIBRATION_IMAGES``, so repeated calls with the same seed always
    produce the same ground truth (while fresh noise is drawn every time).
    """
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    shape = tuple(params["shape"])
    blur_sigma = float(params["blur_sigma"])
    factor = int(params["factor"])
    noise_sigma = float(params["noise_sigma"])
    alpha = float(params["alpha"])

    if image_name is None:
        image_name = CALIBRATION_IMAGES[seed % len(CALIBRATION_IMAGES)]

    x_true = _load_gray(image_name, shape)
    y_clean = blur_downsample(x_true, blur_sigma, factor)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(y_clean.shape) * noise_sigma
    y = y_clean + noise

    return Instance(
        y=y,
        x_true=x_true,
        shape=shape,
        blur_sigma=blur_sigma,
        factor=factor,
        noise_sigma=noise_sigma,
        alpha=alpha,
        image_name=image_name,
        seed=int(seed),
    )


def bicubic_baseline(
    y: np.ndarray,
    shape: tuple[int, int],
    blur_sigma: float,  # noqa: ARG001 -- bicubic ignores the blur; kept for signature symmetry
    factor: int,  # noqa: ARG001 -- inferred from shape / y
    noise_sigma: float,
    **_: Any,
) -> Prediction:
    """Bicubic upsampling with a per-pixel uncertainty estimate.

    The point estimate is a classical bicubic-spline interpolation of the LR
    measurement. ``sigma_hat`` blends a noise-floor term with a local
    gradient-magnitude term — high uncertainty at edges, low in smooth
    regions — so the conformal reward can reward better-calibrated
    spatially-varying uncertainty from downstream solvers.
    """
    x_hat = sktransform.resize(
        y, shape, order=3, anti_aliasing=False, mode="reflect"
    ).astype(np.float64)

    from scipy.ndimage import sobel

    gx = sobel(x_hat, axis=0, mode="reflect")
    gy = sobel(x_hat, axis=1, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy)
    g_max = float(grad.max())
    grad_norm = grad / (g_max + 1e-8)
    sigma_hat = 2.0 * float(noise_sigma) + 0.20 * grad_norm
    sigma_hat = np.maximum(sigma_hat, 1e-4)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


def zero_baseline(
    y: np.ndarray,
    shape: tuple[int, int],
    **_: Any,
) -> Prediction:
    """Trivial lower-bound — predicts the mean of ``y`` everywhere, unit uncertainty."""
    mean_val = float(y.mean())
    x_hat = np.full(shape, mean_val, dtype=np.float64)
    sigma_hat = np.ones(shape, dtype=np.float64) * 0.5
    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


def _score_psnr(x_hat: np.ndarray, x_true: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((x_hat - x_true) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    score = float(np.clip((psnr - PSNR_MIN_DB) / (PSNR_MAX_DB - PSNR_MIN_DB), 0.0, 1.0))
    return score, psnr


def _score_ssim(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    value = ssim_metric(x_true, x_hat, data_range=1.0)
    return float(max(0.0, min(1.0, value)))


class SuperResolutionEnv:
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
        self, seed: int, image_name: str | None = None, **kwargs: Any
    ) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(seed, image_name=image_name, **merged)

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
                "image_name": instance.image_name,
            },
        }

    def run_baseline(
        self, seed: int = 0, image_name: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        instance = self.generate_instance(seed, image_name=image_name, **kwargs)
        prediction = bicubic_baseline(**instance.as_inputs())
        return self.score(prediction, instance)


def calibrate_conformal_quantile(
    alpha: float = DEFAULT_HYPERPARAMS["alpha"],
    n_samples: int | None = None,
    hyperparams: dict[str, Any] | None = None,
    images: tuple[str, ...] = CALIBRATION_IMAGES,
) -> float:
    """Run the bicubic baseline on a fixed rotation of calibration images,
    pool all per-pixel standardized residuals, and return the ``(1 - alpha)``
    split-conformal quantile.

    ``n_samples`` caps the number of images (defaults to all). The rotation
    order is fixed for reproducibility — calibration is deterministic given
    ``hyperparams`` and the image list.
    """
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    names = images[:n_samples] if n_samples is not None else images
    residuals_list = []
    for seed, name in enumerate(names):
        inst = generate_instance(seed=seed, image_name=name, **params)
        pred = bicubic_baseline(**inst.as_inputs())
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
) -> SuperResolutionEnv:
    """Factory. If ``calibration_quantile`` is provided, skip calibration.

    With ``fast=True`` (default) the first call calibrates on the first three
    images of ``CALIBRATION_IMAGES`` (~1 s). With ``fast=False`` all six are
    used. Pass ``calibration_quantile`` explicitly to skip calibration entirely.
    """
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), 3)
    else:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), None)
    return SuperResolutionEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "CALIBRATION_IMAGES",
    "Instance",
    "Prediction",
    "SuperResolutionEnv",
    "generate_instance",
    "bicubic_baseline",
    "zero_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
]
