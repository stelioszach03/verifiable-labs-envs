"""Ground-truth generator for __ENV_ID__.

Each call must produce a fresh ``x*`` instance from the supplied
seed. Procedural regeneration from a 64-bit seed plus a finite
ground-truth pool gives the contamination-resistance guarantee that
makes the env safe to use as an RLVR training signal.

TODO: replace the body of :func:`generate_ground_truth` with your
domain's data source. Examples:
- ``sparse_fourier``: random k-sparse vectors with standard-normal
  amplitudes.
- ``super_resolution``: tile/crop from skimage.data + DIV2K patches.
- ``lodopab_ct``: Shepp-Logan / random-ellipse phantoms or LoDoPaB
  HDF5 slices via ``use_real_data=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Instance:
    """One problem draw. ``x_true`` is the oracle field used by the
    scorer only — solvers must treat it as hidden."""

    y: np.ndarray
    x_true: np.ndarray
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        """Public inputs visible to the solver. Excludes oracle fields."""
        return {
            "y": self.y,
            **self.metadata,
        }


@dataclass(frozen=True)
class Prediction:
    """Solver's answer.

    ``sigma_hat`` carries per-entry uncertainty for the conformal
    reward term; pass ``np.ones_like(x_hat)`` if your solver only
    produces a point estimate (the conformal score will reward the
    resulting wide interval poorly, on purpose).
    """

    x_hat: np.ndarray
    sigma_hat: np.ndarray


def generate_ground_truth(seed: int, **hyperparams: Any) -> np.ndarray:
    """Sample a fresh ``x*`` from the per-env distribution.

    Determinism: two calls with the same seed and hyperparameters
    must return ``np.allclose``-equal results.
    """
    raise NotImplementedError(
        "TODO: implement ground-truth generator for __ENV_ID__. "
        "Use np.random.default_rng(seed) for reproducibility."
    )


__all__ = ["Instance", "Prediction", "generate_ground_truth"]
