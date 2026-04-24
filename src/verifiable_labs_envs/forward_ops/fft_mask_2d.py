"""2D FFT subsampling (MRI k-space undersampling) forward operator."""
from __future__ import annotations

import numpy as np

from verifiable_labs_envs.forward_ops.base import ForwardOperator


class FFTMask2DOp(ForwardOperator):
    """``A(x) = M ⊙ F₂(x)`` — 2D orthonormal DFT with a binary Cartesian mask.

    For MRI knee reconstruction (and any 2D k-space undersampling problem).
    ``x`` is a real or complex image of shape ``(H, W)``; ``mask`` is a
    0/1 array of the same shape whose 1-entries mark retained k-space
    coefficients. The adjoint is mask-then-inverse-DFT ("zero-filled
    reconstruction" is exactly the adjoint applied to the measurement).
    """

    def __init__(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D; got shape {mask.shape}")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0s and 1s")
        self.mask = mask.astype(np.float64)
        self.h, self.w = mask.shape
        self._sampled_count = int(self.mask.sum())

    @property
    def name(self) -> str:
        frac = self._sampled_count / (self.h * self.w) if self.h * self.w else 0.0
        return f"fft_mask_2d(shape={self.h}x{self.w}, keep={frac:.2f})"

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.h, self.w):
            raise ValueError(f"x shape {x.shape} must match mask shape ({self.h}, {self.w})")
        X = np.fft.fft2(x, norm="ortho")
        return X * self.mask  # zero at non-sampled positions

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if y.shape != (self.h, self.w):
            raise ValueError(f"y shape {y.shape} must match mask shape ({self.h}, {self.w})")
        return np.fft.ifft2(y * self.mask, norm="ortho")

    def pseudoinverse(self, y: np.ndarray) -> np.ndarray:
        """Zero-filled reconstruction — the classical MRI baseline."""
        return np.real(self.adjoint(y))

    @staticmethod
    def cartesian_undersample_mask(
        shape: tuple[int, int],
        acceleration: int,
        center_fraction: float = 0.08,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Standard fastMRI-style Cartesian mask: keep a dense center, random rest.

        Returns a (H, W) mask where entire *columns* are either kept or
        dropped (1D undersampling along phase-encode direction, the real
        MRI protocol).
        """
        if acceleration < 1:
            raise ValueError(f"acceleration must be >= 1; got {acceleration}")
        if not (0.0 < center_fraction <= 1.0):
            raise ValueError(f"center_fraction must be in (0, 1]; got {center_fraction}")
        if rng is None:
            rng = np.random.default_rng(0)
        h, w = shape
        n_center = max(1, int(round(w * center_fraction)))
        mask_1d = np.zeros(w, dtype=np.float64)
        center_start = (w - n_center) // 2
        mask_1d[center_start : center_start + n_center] = 1.0
        n_total_keep = max(n_center, w // acceleration)
        remaining_choices = [i for i in range(w) if mask_1d[i] == 0.0]
        n_extra = max(0, n_total_keep - n_center)
        if n_extra > 0 and remaining_choices:
            extras = rng.choice(remaining_choices, size=min(n_extra, len(remaining_choices)), replace=False)
            mask_1d[extras] = 1.0
        # Broadcast along phase-encode rows: every row uses the same column pattern.
        return np.tile(mask_1d[None, :], (h, 1))
