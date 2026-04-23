"""Unit tests for the forward-operator primitives."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs import forward_ops


def test_sparse_fourier_forward_shape() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32)
    mask = np.array([0, 5, 10, 15])
    y = forward_ops.sparse_fourier_forward(x, mask)
    assert y.shape == (4,)
    assert np.iscomplexobj(y)


def test_sparse_fourier_forward_rejects_2d() -> None:
    x = np.zeros((4, 4))
    mask = np.array([0])
    with pytest.raises(ValueError, match="1D"):
        forward_ops.sparse_fourier_forward(x, mask)


def test_sparse_fourier_adjoint_shape() -> None:
    y_sub = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    mask = np.array([0, 5, 10])
    z = forward_ops.sparse_fourier_adjoint(y_sub, mask, n=32)
    assert z.shape == (32,)


def test_sparse_fourier_adjoint_identity_on_diracs() -> None:
    """For a delta in frequency space (one observed freq), A^T returns the corresponding inverse-DFT basis vector."""
    n = 16
    mask = np.array([3])
    y_sub = np.array([1.0 + 0j])
    z = forward_ops.sparse_fourier_adjoint(y_sub, mask, n)
    # expected: (1/sqrt(n)) * exp(+2*pi*i * 3 * n_idx / n)  -- the orthonormal inverse DFT of a single spike
    n_idx = np.arange(n)
    expected = (1.0 / np.sqrt(n)) * np.exp(2j * np.pi * 3 * n_idx / n)
    np.testing.assert_allclose(z, expected, atol=1e-10)


def test_sparse_fourier_adjoint_forward_identity() -> None:
    """``<A x, y> == <x, A^T y>`` — the operator adjoint relation."""
    rng = np.random.default_rng(42)
    n, m = 32, 8
    x = rng.standard_normal(n).astype(np.complex128)
    mask = np.sort(rng.choice(n, size=m, replace=False))
    y = rng.standard_normal(m) + 1j * rng.standard_normal(m)

    ax = forward_ops.sparse_fourier_forward(x, mask)
    aty = forward_ops.sparse_fourier_adjoint(y, mask, n)

    lhs = np.vdot(ax, y)  # <A x, y>
    rhs = np.vdot(x, aty)  # <x, A^T y>
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_sparse_fourier_sample_mask_determinism() -> None:
    mask_a = forward_ops.sparse_fourier_sample_mask(
        n=64, m=16, rng=np.random.default_rng(7)
    )
    mask_b = forward_ops.sparse_fourier_sample_mask(
        n=64, m=16, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(mask_a, mask_b)
    assert mask_a.shape == (16,)
    assert len(np.unique(mask_a)) == 16
    assert mask_a.min() >= 0 and mask_a.max() < 64
    assert np.all(np.diff(mask_a) > 0)  # sorted


def test_sparse_fourier_sample_mask_rejects_m_gt_n() -> None:
    with pytest.raises(ValueError, match="must be <="):
        forward_ops.sparse_fourier_sample_mask(
            n=4, m=10, rng=np.random.default_rng(0)
        )


# ---------- Blur + downsample ----------


def test_blur_downsample_shape() -> None:
    x = np.random.default_rng(0).standard_normal((64, 64))
    y = forward_ops.blur_downsample(x, blur_sigma=1.0, factor=4)
    assert y.shape == (16, 16)


def test_blur_downsample_rejects_bad_shape() -> None:
    x = np.zeros((30, 30))
    with pytest.raises(ValueError, match="divisible"):
        forward_ops.blur_downsample(x, blur_sigma=1.0, factor=4)


def test_blur_downsample_rejects_1d() -> None:
    x = np.zeros(64)
    with pytest.raises(ValueError, match="2D"):
        forward_ops.blur_downsample(x, blur_sigma=1.0, factor=4)


def test_blur_upsample_adjoint_shape() -> None:
    y = np.random.default_rng(0).standard_normal((16, 16))
    z = forward_ops.blur_upsample_adjoint(
        y, blur_sigma=1.0, factor=4, target_shape=(64, 64)
    )
    assert z.shape == (64, 64)


def test_blur_downsample_adjoint_relation() -> None:
    """``<A x, y> == <x, A^T y>`` up to boundary reflection."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 64))
    y = rng.standard_normal((16, 16))
    ax = forward_ops.blur_downsample(x, blur_sigma=1.0, factor=4)
    aty = forward_ops.blur_upsample_adjoint(
        y, blur_sigma=1.0, factor=4, target_shape=(64, 64)
    )
    lhs = float(np.sum(ax * y))
    rhs = float(np.sum(x * aty))
    # Symmetric Gaussian + reflect boundary is self-adjoint up to numeric noise.
    assert abs(lhs - rhs) / (abs(lhs) + 1e-12) < 0.05
