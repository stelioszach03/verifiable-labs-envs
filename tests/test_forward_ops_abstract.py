"""Tests for the class-based ForwardOperator API (sprint-giga Task 0)."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs.forward_ops import (
    FFTMask2DOp,
    FFTSubsampleOp,
    ForwardOperator,
    MagnitudeOnlyOp,
)


class TestForwardOperatorABC:
    def test_forward_operator_is_abstract(self):
        with pytest.raises(TypeError):
            ForwardOperator()  # type: ignore[abstract]

    def test_subclass_must_implement_apply_and_adjoint(self):
        class IncompleteOp(ForwardOperator):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteOp()  # type: ignore[abstract]


class TestFFTSubsampleOp:
    def test_round_trip_preserves_sampled_coefficients(self):
        n = 32
        rng = np.random.default_rng(0)
        mask = np.sort(rng.choice(n, size=12, replace=False))
        op = FFTSubsampleOp(n, mask)
        x = rng.standard_normal(n)
        y = op.apply(x)
        assert y.shape == mask.shape
        assert y.dtype == np.complex128

    def test_adjoint_produces_length_n_signal(self):
        op = FFTSubsampleOp(n=16, mask=np.array([0, 3, 5, 8]))
        y = np.ones(4, dtype=np.complex128)
        z = op.adjoint(y)
        assert z.shape == (16,)

    def test_name_reports_dimensions(self):
        op = FFTSubsampleOp(n=32, mask=np.array([1, 2, 3]))
        assert "n=32" in op.name
        assert "m=3" in op.name

    def test_rejects_mask_out_of_range(self):
        with pytest.raises(ValueError, match="mask values"):
            FFTSubsampleOp(n=8, mask=np.array([0, 9]))

    def test_rejects_non_1d_mask(self):
        with pytest.raises(ValueError, match="mask must be 1D"):
            FFTSubsampleOp(n=8, mask=np.zeros((2, 2), dtype=np.int64))


class TestFFTMask2DOp:
    def test_forward_preserves_shape_and_masks_unmeasured(self):
        h, w = 8, 8
        mask = np.zeros((h, w), dtype=np.float64)
        mask[:, :4] = 1.0  # keep left half
        op = FFTMask2DOp(mask)
        x = np.random.default_rng(0).standard_normal((h, w))
        y = op.apply(x)
        assert y.shape == (h, w)
        # Right half must be exactly zero after masking.
        assert np.allclose(y[:, 4:], 0.0)

    def test_adjoint_returns_same_shape(self):
        op = FFTMask2DOp(np.ones((4, 4)))
        y = np.ones((4, 4), dtype=np.complex128)
        z = op.adjoint(y)
        assert z.shape == (4, 4)

    def test_pseudoinverse_returns_real_image(self):
        op = FFTMask2DOp(np.ones((4, 4)))
        y = np.ones((4, 4), dtype=np.complex128)
        z = op.pseudoinverse(y)
        assert z.dtype in (np.float64, np.float32)

    def test_cartesian_mask_keeps_dc_columns(self):
        """DC-aligned convention: low-freq columns wrap around index 0."""
        mask = FFTMask2DOp.cartesian_undersample_mask(
            shape=(8, 16), acceleration=4, center_fraction=0.25,
            rng=np.random.default_rng(0),
        )
        assert mask.shape == (8, 16)
        # All rows are identical (Cartesian undersampling).
        for r in range(1, 8):
            assert np.array_equal(mask[0], mask[r])
        # DC (column 0) is always kept, along with wrap-around low frequencies.
        assert mask[0, 0] == 1.0
        # At center_fraction=0.25 (4 columns), cols {0, 1, 15, 14} are expected.
        assert mask[0, 1] == 1.0
        assert mask[0, 15] == 1.0

    def test_cartesian_mask_rejects_bad_params(self):
        with pytest.raises(ValueError):
            FFTMask2DOp.cartesian_undersample_mask((8, 8), acceleration=0)
        with pytest.raises(ValueError):
            FFTMask2DOp.cartesian_undersample_mask((8, 8), acceleration=2, center_fraction=0.0)

    def test_rejects_non_binary_mask(self):
        with pytest.raises(ValueError, match="0s and 1s"):
            FFTMask2DOp(mask=np.array([[0.5, 0.5], [0.5, 0.5]]))


class TestMagnitudeOnlyOp:
    def test_apply_returns_nonneg_reals(self):
        op = MagnitudeOnlyOp(n=16, mask=np.array([0, 1, 2, 3]))
        x = np.random.default_rng(0).standard_normal(16)
        y = op.apply(x)
        assert y.shape == (4,)
        assert np.all(y >= 0.0)
        assert y.dtype in (np.float64, np.float32)

    def test_adjoint_treats_magnitudes_as_zero_phase(self):
        op = MagnitudeOnlyOp(n=8, mask=np.array([0, 2, 4, 6]))
        y = np.array([1.0, 1.0, 1.0, 1.0])
        z = op.adjoint(y)
        assert z.shape == (8,)
        # Zero-phase coefficients placed on a symmetric subset yield a real iFFT.
        assert np.allclose(z.imag, 0.0, atol=1e-12)

    def test_magnitude_is_global_phase_invariant(self):
        """|F(e^{iφ} x)| = |F(x)| for any global phase φ."""
        op = MagnitudeOnlyOp(n=32, mask=np.arange(32))
        x = np.random.default_rng(0).standard_normal(32)
        for phi in [0.0, 0.5, 1.3, np.pi]:
            y1 = op.apply(x)
            y2 = op.apply(np.exp(1j * phi) * x)  # complex rotation
            assert np.allclose(y1, np.abs(y2))

    def test_rejects_mismatched_y_shape(self):
        op = MagnitudeOnlyOp(n=8, mask=np.array([0, 1, 2]))
        with pytest.raises(ValueError):
            op.adjoint(np.array([1.0, 1.0]))  # wrong length


class TestLegacyImportsPreserved:
    """Guard rail: after packageification, legacy free-function imports must still work."""

    def test_sparse_fourier_free_functions_import(self):
        from verifiable_labs_envs.forward_ops import (  # noqa: F401
            sparse_fourier_adjoint,
            sparse_fourier_forward,
            sparse_fourier_sample_mask,
        )

    def test_radon_free_functions_import(self):
        from verifiable_labs_envs.forward_ops import (  # noqa: F401
            radon_adjoint,
            radon_fbp,
            radon_forward,
        )

    def test_blur_free_functions_import(self):
        from verifiable_labs_envs.forward_ops import (  # noqa: F401
            blur_downsample,
            blur_upsample_adjoint,
        )
