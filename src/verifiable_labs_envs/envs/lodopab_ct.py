"""Environment 3 — low-dose 2D parallel-beam CT reconstruction.

    y = R(x*) + eta,    eta ~ N(0, sigma^2 I)

where ``R`` is the 2D parallel-beam Radon transform at ``n_angles`` angles (4x
undersampled relative to the full-angular Nyquist rate for this image size).

**Two ground-truth modes.**

Default is ``use_real_data=False``: ``x*`` is drawn from a small rotation of
synthetic phantoms (Shepp-Logan plus four skimage test images used as
attenuation stand-ins, each circularly masked). Safe for CI because no
download is required.

With ``use_real_data=True`` (requires a one-time ~350 MB download via
``scripts/download_lodopab_validation.sh``) ``x*`` is drawn from the 3552
validation slices of the LoDoPaB-CT dataset (Leuschner et al. 2021, Nature
Scientific Data), which are real clinical chest-CT reconstructions from the
LIDC-IDRI cohort. The Radon geometry is still parallel-beam (matching
LoDoPaB's published spec) so the forward operator is unchanged.

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
from pathlib import Path
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA_DIR: Path = _REPO_ROOT / "data" / "lodopab_ct"
REAL_DATA_N_SLICES: int = 3552  # published LoDoPaB validation partition size
REAL_DATA_SLICES_PER_CHUNK: int = 128  # LoDoPaB ships validation as chunks of 128 slices
REAL_DATA_HDF5_KEY: str = "data"  # LoDoPaB convention: dataset named "data"


def _real_data_chunks() -> list[Path]:
    """Sorted list of ``ground_truth_validation_NNN.hdf5`` chunk files on disk."""
    if not REAL_DATA_DIR.exists():
        return []
    return sorted(REAL_DATA_DIR.glob("ground_truth_validation_*.hdf5"))


def has_real_data() -> bool:
    """True iff at least one LoDoPaB-CT validation chunk HDF5 is on disk."""
    return len(_real_data_chunks()) > 0

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


def _apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """Zero everything outside the inscribed disc (required by parallel-beam Radon)."""
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 1.0
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return (img * mask).astype(np.float64)


@lru_cache(maxsize=128)
def _load_real_slice(index: int, shape: tuple[int, int]) -> np.ndarray:
    """Load a real LoDoPaB-CT validation slice, resized to ``shape``, [0, 1]-normalized, disc-masked.

    LoDoPaB ships the validation partition as a sequence of chunk HDF5 files
    (``ground_truth_validation_000.hdf5``, ...) each containing 128 slices
    under key ``"data"``. This function maps a flat slice index to
    ``(chunk_id, within_chunk)``, opens the correct HDF5 file lazily, reads
    the slice, and applies the same normalization + circular-mask pipeline
    as the phantom path. ``h5py`` is imported lazily so the default CI path
    (phantom only) does not pull an extra dependency.
    """
    chunks = _real_data_chunks()
    if not chunks:
        raise FileNotFoundError(
            f"LoDoPaB-CT validation data not found under {REAL_DATA_DIR}. "
            "Run `bash scripts/download_lodopab_validation.sh` once to fetch "
            "and unpack the ~1.5 GB ground-truth ZIP from Zenodo."
        )
    chunk_id, within_chunk = divmod(int(index), REAL_DATA_SLICES_PER_CHUNK)
    if chunk_id >= len(chunks):
        raise IndexError(
            f"slice index {index} falls in chunk {chunk_id} but only "
            f"{len(chunks)} chunks are on disk"
        )

    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "h5py is required for real-data mode. "
            "Install with `pip install -e '.[ct-real]'`."
        ) from exc

    with h5py.File(chunks[chunk_id], "r") as fh:
        dataset = fh[REAL_DATA_HDF5_KEY]
        n_in_chunk = dataset.shape[0]
        if within_chunk >= n_in_chunk:
            raise IndexError(
                f"slice {index} (chunk {chunk_id}, offset {within_chunk}) "
                f"exceeds chunk size {n_in_chunk}"
            )
        img = np.asarray(dataset[within_chunk], dtype=np.float64)

    img = sktransform.resize(img, shape, anti_aliasing=True, mode="reflect")
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return _apply_circular_mask(img)


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
    return _apply_circular_mask(img)


def _angles_for(n_angles: int) -> np.ndarray:
    """Equispaced projection angles in [0, 180) degrees."""
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def generate_instance(
    seed: int,
    phantom_name: str | None = None,
    use_real_data: bool = False,
    **kwargs: Any,
) -> Instance:
    """Generate one CT reconstruction instance.

    ``use_real_data=False`` (default): ``seed % len(CALIBRATION_PHANTOMS)`` picks
    a synthetic phantom; fresh sinogram noise is drawn from ``seed``.

    ``use_real_data=True``: ``seed % REAL_DATA_N_SLICES`` picks a LoDoPaB-CT
    validation slice; fresh sinogram noise is drawn from ``seed``. Requires
    ``scripts/download_lodopab_validation.sh`` to have been run once.
    """
    params = {**DEFAULT_HYPERPARAMS, **kwargs}
    shape = tuple(params["shape"])
    n_angles = int(params["n_angles"])
    noise_sigma = float(params["noise_sigma"])
    alpha = float(params["alpha"])

    if use_real_data:
        if phantom_name is not None:
            raise ValueError(
                "phantom_name cannot be used with use_real_data=True; "
                "the seed alone picks the validation slice index."
            )
        slice_index = int(seed) % REAL_DATA_N_SLICES
        x_true = _load_real_slice(slice_index, shape)
        source_name = f"lodopab_val_{slice_index}"
    else:
        if phantom_name is None:
            phantom_name = CALIBRATION_PHANTOMS[seed % len(CALIBRATION_PHANTOMS)]
        x_true = _load_phantom(phantom_name, shape)
        source_name = phantom_name
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
        phantom_name=source_name,
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
        use_real_data: bool = False,
    ) -> None:
        self.conformal_quantile = float(conformal_quantile)
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.use_real_data = bool(use_real_data)

    def generate_instance(
        self, seed: int, phantom_name: str | None = None, **kwargs: Any
    ) -> Instance:
        merged = {**self.hyperparams, **kwargs}
        return generate_instance(
            seed,
            phantom_name=phantom_name,
            use_real_data=self.use_real_data,
            **merged,
        )

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
    use_real_data: bool = False,
) -> LodopabCtEnv:
    """Factory. Fast calibration uses the first three phantoms; full uses all five.

    Calibration is always performed on synthetic phantoms (deterministic,
    no network). Real-data mode only affects the evaluation-time
    ``generate_instance`` path.
    """
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), 3)
    else:
        q = _cached_quantile(float(DEFAULT_HYPERPARAMS["alpha"]), None)
    return LodopabCtEnv(conformal_quantile=q, use_real_data=use_real_data)


__all__ = [
    "NAME",
    "DEFAULT_HYPERPARAMS",
    "DEFAULT_WEIGHTS",
    "CALIBRATION_PHANTOMS",
    "REAL_DATA_DIR",
    "REAL_DATA_N_SLICES",
    "Instance",
    "Prediction",
    "LodopabCtEnv",
    "generate_instance",
    "fbp_baseline",
    "zero_baseline",
    "calibrate_conformal_quantile",
    "load_environment",
    "has_real_data",
]
