"""Forward operators shared across inverse-problem environments.

Each operator ``A`` and its adjoint ``A^T`` are exposed as plain NumPy functions,
parametrized by whatever structural inputs the operator needs (e.g. a subsampling
mask, a blur kernel, a Radon projection geometry). Environments use these to (a)
synthesize measurements ``y = A(x) + eta`` with known ground truth and (b) expose
them to solvers so the model is required to actually invert physics rather than
pattern-match on text answers.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def sparse_fourier_forward(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``A(x) = S . F(x)`` — subsampled orthonormal 1D DFT.

    ``x`` is a real- or complex-valued signal of length ``n``. ``mask`` is an
    integer array of observed frequency indices with values in ``[0, n)``.
    Returns complex measurements of shape ``(len(mask),)``.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D; got shape {x.shape}")
    X = np.fft.fft(x, norm="ortho")
    return X[mask]


def sparse_fourier_adjoint(y_sub: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """``A^T(y) = F^* . S^T(y)`` — zero-fill to length ``n`` then inverse orthonormal DFT.

    Returns a complex-valued signal of shape ``(n,)``.
    """
    if y_sub.ndim != 1:
        raise ValueError(f"y_sub must be 1D; got shape {y_sub.shape}")
    if mask.shape != y_sub.shape:
        raise ValueError(f"mask shape {mask.shape} must match y_sub shape {y_sub.shape}")
    z = np.zeros(n, dtype=np.complex128)
    z[mask] = y_sub
    return np.fft.ifft(z, norm="ortho")


def sparse_fourier_sample_mask(
    n: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``m`` of ``n`` frequency indices uniformly without replacement, sorted."""
    if m < 0 or n <= 0:
        raise ValueError(f"expected n > 0 and m >= 0; got n={n}, m={m}")
    if m > n:
        raise ValueError(f"m={m} must be <= n={n}")
    return np.sort(rng.choice(n, size=m, replace=False))


def blur_downsample(
    x: np.ndarray, blur_sigma: float, factor: int
) -> np.ndarray:
    """``A(x) = S . G(x)`` — 2D Gaussian blur followed by stride-``factor`` decimation.

    Mirror-padded convolution (``mode='reflect'``) makes the boundary adjoint
    clean. ``x`` must be 2D with each side divisible by ``factor``; returns an
    array of shape ``(H // factor, W // factor)``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D; got shape {x.shape}")
    if factor < 1:
        raise ValueError(f"factor must be >= 1; got {factor}")
    if x.shape[0] % factor != 0 or x.shape[1] % factor != 0:
        raise ValueError(
            f"each side of x ({x.shape}) must be divisible by factor={factor}"
        )
    blurred = gaussian_filter(x, sigma=blur_sigma, mode="reflect")
    return blurred[::factor, ::factor]


def blur_upsample_adjoint(
    y: np.ndarray, blur_sigma: float, factor: int, target_shape: tuple[int, int]
) -> np.ndarray:
    """``A^T(y) = G . S^T(y)`` — zero-fill insertion at stride ``factor`` then blur.

    Adjoint of :func:`blur_downsample`. Since the Gaussian kernel is symmetric
    the blur is self-adjoint up to the boundary mode; with ``mode='reflect'``
    the operator is its own transpose in the interior.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D; got shape {y.shape}")
    h, w = target_shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"target_shape {target_shape} must be divisible by factor={factor}"
        )
    if y.shape != (h // factor, w // factor):
        raise ValueError(
            f"y shape {y.shape} incompatible with target_shape {target_shape} at factor {factor}"
        )
    upsampled = np.zeros((h, w), dtype=y.dtype)
    upsampled[::factor, ::factor] = y
    return gaussian_filter(upsampled, sigma=blur_sigma, mode="reflect")


def radon_forward(x: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """``A(x)`` — 2D parallel-beam Radon transform at ``angles_deg`` (degrees).

    Returns a sinogram of shape ``(n_side, n_angles)`` where ``n_side`` equals
    the input side length (with ``circle=True`` the useful signal is contained
    in a disc inscribed in the square image).
    """
    from skimage.transform import radon  # lazy import

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"x must be square 2D; got shape {x.shape}")
    return radon(x, theta=angles_deg, circle=True)


def radon_adjoint(
    sinogram: np.ndarray, angles_deg: np.ndarray, output_size: int
) -> np.ndarray:
    """``A^T(y)`` — unfiltered back-projection (the adjoint of Radon).

    Computed as ``iradon(..., filter_name=None)`` which is the sum of
    projections back along each ray, without the Ram-Lak sharpening filter.
    """
    from skimage.transform import iradon  # lazy import

    return iradon(
        sinogram,
        theta=angles_deg,
        output_size=output_size,
        filter_name=None,
        circle=True,
    )


def radon_fbp(
    sinogram: np.ndarray, angles_deg: np.ndarray, output_size: int
) -> np.ndarray:
    """Filtered back-projection with the Ram-Lak filter — the reference CT baseline."""
    from skimage.transform import iradon  # lazy import

    return iradon(
        sinogram,
        theta=angles_deg,
        output_size=output_size,
        filter_name="ramp",
        circle=True,
    )


__all__ = [
    "sparse_fourier_forward",
    "sparse_fourier_adjoint",
    "sparse_fourier_sample_mask",
    "blur_downsample",
    "blur_upsample_adjoint",
    "radon_forward",
    "radon_adjoint",
    "radon_fbp",
]
