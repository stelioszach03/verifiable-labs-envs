"""Forward operators for inverse-problem environments.

Two APIs coexist for back-compat:

- **Free functions** (pre-sprint-giga layout) are re-exported from ``_legacy``
  so every existing import like
  ``from verifiable_labs_envs.forward_ops import sparse_fourier_forward`` still
  resolves.

- **Class-based API** (``ForwardOperator`` subclasses) is the new pattern for
  the four sprint-giga envs (phase retrieval, MRI, seismic, retrosynthesis).
  Each concrete class packages ``apply`` / ``adjoint`` / ``pseudoinverse`` plus
  whatever structural state the operator needs.

New envs should use the class-based API. Existing envs continue to use the
free functions; migrating them is a background task, not a blocker.
"""
from __future__ import annotations

# Back-compat: re-export every legacy free function + constant.
from verifiable_labs_envs.forward_ops._legacy import (
    blur_downsample,
    blur_upsample_adjoint,
    radon_adjoint,
    radon_fbp,
    radon_forward,
    sparse_fourier_adjoint,
    sparse_fourier_forward,
    sparse_fourier_sample_mask,
)

# Class-based API
from verifiable_labs_envs.forward_ops.base import ForwardOperator
from verifiable_labs_envs.forward_ops.fft_mask_2d import FFTMask2DOp
from verifiable_labs_envs.forward_ops.fft_subsample import FFTSubsampleOp
from verifiable_labs_envs.forward_ops.magnitude_only import MagnitudeOnlyOp

__all__ = [
    # legacy free functions (back-compat — do not remove)
    "sparse_fourier_forward",
    "sparse_fourier_adjoint",
    "sparse_fourier_sample_mask",
    "blur_downsample",
    "blur_upsample_adjoint",
    "radon_forward",
    "radon_adjoint",
    "radon_fbp",
    # class-based API
    "ForwardOperator",
    "FFTSubsampleOp",
    "FFTMask2DOp",
    "MagnitudeOnlyOp",
]
