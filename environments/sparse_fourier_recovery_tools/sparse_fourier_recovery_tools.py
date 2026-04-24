"""Tool-use sparse Fourier recovery — primitive composition (fft, ifft, threshold, compute_residual, sparsity_norm). No solver oracle.

Prime Intellect Hub wrapper around ``verifiable_labs_envs.envs.sparse_fourier_tools``.
The monorepo at https://github.com/stelioszach03/verifiable-labs-envs is the
source of truth; this file is a thin re-export so the env can be installed
and discovered via the Prime Intellect Environments Hub.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.sparse_fourier_tools import load_environment as _le


def load_environment(**kwargs: Any):
    """Factory for the ``sparse-fourier-recovery-tools`` environment.

    Passes kwargs through to the monorepo's ``load_environment`` (accepts
    ``calibration_quantile``, ``fast``, and env-specific options like
    ``max_turns`` or ``use_real_data`` where applicable).
    """
    return _le(**kwargs)
