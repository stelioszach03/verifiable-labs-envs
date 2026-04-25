"""Forward operator for __ENV_ID__.

The forward operator ``A`` maps ground-truth ``x*`` to a measurement
``y = A(x*) + noise``. Required to be deterministic given a fixed
seed (the env injects fresh noise per call from the seed for
procedural regeneration).

TODO: replace the body of :func:`forward` with your physics. Examples
in the existing envs:
- ``verifiable_labs_envs.envs.sparse_fourier``: ``y = S · F(x)``
  (subsampled DFT).
- ``verifiable_labs_envs.envs.lodopab_ct``: ``y = R(x)`` (Radon
  transform).
- ``verifiable_labs_envs.envs.mri_knee``: ``y = M ⊙ F₂(x)``
  (Cartesian-undersampled 2D DFT).
"""
from __future__ import annotations

import numpy as np


def forward(x: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Apply the forward operator ``A`` to ``x``.

    Parameters
    ----------
    x : np.ndarray
        Ground-truth signal / image / volume.
    seed : int, optional
        Used to seed any randomness inside the operator (e.g. mask
        sampling). The env passes its own seed through.

    Returns
    -------
    np.ndarray
        Measurement ``y``.
    """
    raise NotImplementedError(
        "TODO: implement forward operator for __ENV_ID__. "
        "See module docstring for examples."
    )


def adjoint(y: np.ndarray) -> np.ndarray:
    """Apply ``A^T`` (or pseudo-inverse) to ``y`` for gradient methods.

    Override only if your domain needs explicit adjoint computation;
    otherwise the env's ``run_baseline`` can use ``np.linalg.lstsq``
    on a small problem.
    """
    raise NotImplementedError(
        "TODO: implement adjoint operator for __ENV_ID__ "
        "(or delete if not needed)."
    )


__all__ = ["forward", "adjoint"]
