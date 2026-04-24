"""Class wrapper over the 1D subsampled-Fourier free functions."""
from __future__ import annotations

import numpy as np

from verifiable_labs_envs.forward_ops._legacy import (
    sparse_fourier_adjoint,
    sparse_fourier_forward,
)
from verifiable_labs_envs.forward_ops.base import ForwardOperator


class FFTSubsampleOp(ForwardOperator):
    """``A(x) = S . F(x)`` — 1D orthonormal DFT subsampled by ``mask``.

    Used by ``sparse-fourier-recovery`` and its multi-turn / tool-use
    variants. This class is a thin wrapper over the legacy free functions
    in ``forward_ops._legacy``.
    """

    def __init__(self, n: int, mask: np.ndarray) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0; got {n}")
        mask = np.asarray(mask, dtype=np.int64)
        if mask.ndim != 1:
            raise ValueError(f"mask must be 1D; got shape {mask.shape}")
        if mask.size and (mask.min() < 0 or mask.max() >= n):
            raise ValueError(f"mask values must be in [0, {n}); got min={mask.min()}, max={mask.max()}")
        self.n = int(n)
        self.mask = mask

    @property
    def name(self) -> str:
        return f"fft_subsample(n={self.n}, m={self.mask.size})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return sparse_fourier_forward(x, self.mask)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return sparse_fourier_adjoint(y, self.mask, self.n)
