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


__all__ = [
    "sparse_fourier_forward",
    "sparse_fourier_adjoint",
    "sparse_fourier_sample_mask",
]
