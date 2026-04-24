"""Magnitude-only Fourier forward operator (phase retrieval)."""
from __future__ import annotations

import numpy as np

from verifiable_labs_envs.forward_ops.base import ForwardOperator


class MagnitudeOnlyOp(ForwardOperator):
    """``A(x) = |F(x)|`` — magnitude of the 1D subsampled DFT.

    This is *non-linear*: the magnitude breaks the linearity of ``F``.
    Phase retrieval is the problem of recovering ``x`` from ``|F(x)|``,
    modulo the inherent global-phase / flip / shift ambiguities.

    - ``apply(x)`` returns real non-negative magnitudes of length ``mask.size``.
    - ``adjoint(y)`` is not a true linear adjoint; we return the dirty-
      image iFFT of ``y`` treated as a zero-phase complex signal. This is
      the "magnitude-as-real" heuristic often used to initialize
      Gerchberg-Saxton.
    - ``pseudoinverse(y)`` is the same dirty-image, kept for API symmetry.
    """

    def __init__(self, n: int, mask: np.ndarray) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0; got {n}")
        mask = np.asarray(mask, dtype=np.int64)
        if mask.ndim != 1:
            raise ValueError(f"mask must be 1D; got shape {mask.shape}")
        if mask.size and (mask.min() < 0 or mask.max() >= n):
            raise ValueError(
                f"mask values must be in [0, {n}); got min={mask.min()}, max={mask.max()}"
            )
        self.n = int(n)
        self.mask = mask

    @property
    def name(self) -> str:
        return f"magnitude_only(n={self.n}, m={self.mask.size})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"x must be 1D; got shape {x.shape}")
        X = np.fft.fft(x, norm="ortho")
        return np.abs(X[self.mask])

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Treat magnitudes as zero-phase coefficients → dirty-image iFFT.

        Not the Jacobian adjoint; this is a heuristic suitable for
        Gerchberg-Saxton initialization.
        """
        if y.ndim != 1 or y.shape != self.mask.shape:
            raise ValueError(
                f"y shape {y.shape} must match mask shape {self.mask.shape}"
            )
        z = np.zeros(self.n, dtype=np.complex128)
        z[self.mask] = y.astype(np.complex128)
        return np.fft.ifft(z, norm="ortho")
