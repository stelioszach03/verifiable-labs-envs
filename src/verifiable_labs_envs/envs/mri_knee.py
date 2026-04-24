"""Environment — MRI knee reconstruction from undersampled k-space.

Accelerated MRI is the canonical medical-imaging inverse problem: recover an
MR image from an undersampled k-space acquisition, where undersampling is
along the phase-encode direction (Cartesian protocol, 4× acceleration is
clinical-standard).

Problem:
    y = M ⊙ F₂(x*) + eta,  eta ~ N(0, sigma^2 I) (in k-space)

with ``F₂`` the 2D orthonormal DFT, ``M`` a binary Cartesian mask (dense
center-of-k-space + random outer columns), ``x* ∈ R^{h×w}`` the ground-truth
image.

Rather than requiring a fastMRI dataset download (gated by NYU application),
v1 synthesizes ground truth from ``skimage.data`` images resized to the
target resolution and grayscale-normalized. This keeps the env fully
self-contained; fastMRI integration is a v2 follow-up documented in
``docs/MRI_DATA.md``.

Image resolution is kept at 16×16 for LLM tractability (256 pixels →
tractable JSON size with int-in-[0,255] encoding). Reward components:

  * PSNR — clamped to [15 dB → 0, 35 dB → 1].
  * SSIM — window-size-3 (small-image friendly), clipped to [0, 1].
  * Conformal coverage — split-conformal on zero-filled-reconstruction
    residuals.

Ambiguities: none — the forward operator is linear with a known mask, so
recovery is unique modulo noise.
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
from verifiable_labs_envs.forward_ops import FFTMask2DOp

NAME = "mri-knee-reconstruction"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "shape": (16, 16),
    "acceleration": 4,  # 4x undersampling
    "center_fraction": 0.25,  # keep 25% center of k-space
    "noise_sigma": 0.02,  # k-space-domain
    "alpha": 0.1,
}

DEFAULT_WEIGHTS: dict[str, float] = {"psnr": 0.4, "ssim": 0.3, "conformal": 0.3}

# Rotation of public-domain skimage images used as synthetic MRI ground truth.
# Resized + normalized to the instance shape at generation time.
CALIBRATION_IMAGES: tuple[str, ...] = (
    "camera", "moon", "astronaut", "coffee", "chelsea", "immunohistochemistry",
)

PSNR_MIN_DB: float = 15.0
PSNR_MAX_DB: float = 35.0


@dataclass(frozen=True)
class Instance:
    y: np.ndarray  # complex k-space (zero at non-sampled positions), shape=(h,w)
    mask: np.ndarray  # 0/1 float, shape=(h,w)
    zero_filled: np.ndarray  # real-valued zero-filled IFFT reconstruction, shape=(h,w)
    shape: tuple[int, int]
    noise_sigma: float
    alpha: float
    x_true: np.ndarray  # real, shape=(h,w), in [0,1]
    seed: int

    def as_inputs(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "mask": self.mask,
            "shape": self.shape,
            "noise_sigma": self.noise_sigma,
        }


@dataclass(frozen=True)
class Prediction:
    x_hat: np.ndarray  # real image shape=(h,w) in [0,1]
    sigma_hat: np.ndarray  # shape=(h,w), strictly positive


def _load_ground_truth_image(seed: int, shape: tuple[int, int]) -> np.ndarray:
    """Pick a skimage image, convert to grayscale, resize to shape, normalize [0,1]."""
    img_name = CALIBRATION_IMAGES[seed % len(CALIBRATION_IMAGES)]
    loader = getattr(skdata, img_name)
    img = loader()
    if img.ndim == 3:
        img = img.mean(axis=-1)  # grayscale
    img = img.astype(np.float64)
    if img.max() > 1.01:
        img = img / 255.0
    resized = sktransform.resize(img, shape, order=3, anti_aliasing=True, mode="reflect")
    return np.clip(resized, 0.0, 1.0)


def generate_instance(seed: int, **kwargs: Any) -> Instance:
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    shape = tuple(params["shape"])
    acceleration = int(params["acceleration"])
    center_fraction = float(params["center_fraction"])
    noise_sigma = float(params["noise_sigma"])
    alpha = float(params["alpha"])

    rng = np.random.default_rng(seed)
    x_true = _load_ground_truth_image(seed, shape)

    mask = FFTMask2DOp.cartesian_undersample_mask(
        shape=shape, acceleration=acceleration, center_fraction=center_fraction,
        rng=rng,
    )
    op = FFTMask2DOp(mask)
    y_clean = op.apply(x_true)
    # k-space noise (complex Gaussian)
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * (noise_sigma / np.sqrt(2.0))
    y = y_clean + noise * mask  # noise only at sampled positions

    zero_filled = op.pseudoinverse(y)  # real-valued zero-filled IFFT

    return Instance(
        y=y,
        mask=mask,
        zero_filled=zero_filled,
        shape=shape,
        noise_sigma=noise_sigma,
        alpha=alpha,
        x_true=x_true,
        seed=int(seed),
    )


def zero_filled_baseline(
    y: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    noise_sigma: float,
    **_: Any,
) -> Prediction:
    """Zero-filled inverse FFT — classical MRI baseline."""
    op = FFTMask2DOp(mask)
    x_hat = np.clip(op.pseudoinverse(y), 0.0, 1.0)
    # Approximate sigma_hat: uniform noise_sigma scaled by sqrt(1/acceleration)
    sigma_hat = np.full(shape, max(float(noise_sigma) * 2.0, 0.02), dtype=np.float64)
    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


def tv_regularized_baseline(
    y: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    noise_sigma: float,
    n_iters: int = 50,
    tv_weight: float = 0.01,
    **_: Any,
) -> Prediction:
    """Simple gradient-descent with TV regularization — stronger baseline than zero-filled.

    Minimizes ``||M·F(x) - y||^2 + lambda·TV(x)`` via projected gradient.
    """
    op = FFTMask2DOp(mask)
    # Warm-start from zero-filled
    x = np.clip(op.pseudoinverse(y), 0.0, 1.0)
    step = 0.5

    for _ in range(n_iters):
        # Data fidelity gradient: A^T (Ax - y)
        Ax = op.apply(x)
        grad_data = np.real(op.adjoint(Ax - y))
        # TV gradient (isotropic)
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:-1, :] = np.diff(x, axis=0)
        gy[:, :-1] = np.diff(x, axis=1)
        mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        div_x = np.zeros_like(x)
        div_y = np.zeros_like(x)
        div_x[1:, :] = gx[:-1, :] / mag[:-1, :]
        div_x[:-1, :] -= gx[:-1, :] / mag[:-1, :]
        div_y[:, 1:] = gy[:, :-1] / mag[:, :-1]
        div_y[:, :-1] -= gy[:, :-1] / mag[:, :-1]
        grad_tv = -(div_x + div_y)
        x = x - step * (grad_data + tv_weight * grad_tv)
        x = np.clip(x, 0.0, 1.0)

    sigma_hat = np.full(shape, max(float(noise_sigma) * 2.0, 0.02), dtype=np.float64)
    return Prediction(x_hat=x, sigma_hat=sigma_hat)


def _psnr_db(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    mse = float(np.mean((x_true - x_hat) ** 2))
    if mse <= 1e-12:
        return 60.0
    data_range = 1.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _psnr_score(psnr_db: float) -> float:
    x = (psnr_db - PSNR_MIN_DB) / (PSNR_MAX_DB - PSNR_MIN_DB)
    return float(np.clip(x, 0.0, 1.0))


class MRIKneeEnv:
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
        x_hat = np.clip(prediction.x_hat, 0.0, 1.0)

        psnr = _psnr_db(instance.x_true, x_hat)
        score_psnr = _psnr_score(psnr)
        # small window for small images
        ws = min(3, min(instance.shape) if min(instance.shape) % 2 == 1 else min(instance.shape) - 1)
        if ws < 3:
            ws = 3
        try:
            score_ssim = float(np.clip(
                ssim_metric(instance.x_true, x_hat, data_range=1.0, win_size=ws),
                0.0, 1.0,
            ))
        except Exception:  # noqa: BLE001
            # Fallback if SSIM complains about window size vs image size.
            score_ssim = max(0.0, 1.0 - float(np.mean(np.abs(instance.x_true - x_hat))))

        # Conformal coverage: use full-image residuals (small images, so
        # zero-dominance is less extreme than sparse-F).
        lo, hi = interval(x_hat, prediction.sigma_hat, self.conformal_quantile)
        cov = coverage(instance.x_true.ravel(), lo.ravel(), hi.ravel())
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
                "psnr_db": psnr,
                "coverage": cov,
                "target_coverage": target,
                "conformal_quantile": self.conformal_quantile,
                "weights": dict(w),
            },
        }

    def run_baseline(self, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Zero-filled-IFFT baseline — canonical MRI reference reconstruction."""
        instance = self.generate_instance(seed, **kwargs)
        prediction = zero_filled_baseline(**instance.as_inputs())
        return self.score(prediction, instance)


def calibrate_conformal_quantile(
    n_samples: int = 100,
    alpha: float = DEFAULT_HYPERPARAMS["alpha"],
    hyperparams: dict[str, Any] | None = None,
    start_seed: int = 10_000,
) -> float:
    """Split-conformal on zero-filled-IFFT residuals."""
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    residuals_list = []
    for i in range(n_samples):
        inst = generate_instance(seed=start_seed + i, **params)
        pred = zero_filled_baseline(**inst.as_inputs())
        r = scaled_residuals(pred.x_hat, inst.x_true, pred.sigma_hat)
        residuals_list.append(r.ravel())
    pooled = np.concatenate(residuals_list)
    return split_conformal_quantile(pooled, alpha)


@lru_cache(maxsize=8)
def _cached_quantile(n_samples: int, alpha: float) -> float:
    return calibrate_conformal_quantile(n_samples=n_samples, alpha=alpha)


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
) -> MRIKneeEnv:
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(n_samples=20, alpha=float(DEFAULT_HYPERPARAMS["alpha"]))
    else:
        q = _cached_quantile(n_samples=100, alpha=float(DEFAULT_HYPERPARAMS["alpha"]))
    return MRIKneeEnv(conformal_quantile=q)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "Instance",
    "Prediction",
    "MRIKneeEnv",
    "generate_instance",
    "zero_filled_baseline",
    "tv_regularized_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
]
