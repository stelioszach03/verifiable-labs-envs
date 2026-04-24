"""Abstract base class for all forward operators used in inverse-problem envs."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForwardOperator(ABC):
    """A linear (or nonlinear) measurement operator ``y = A(x)``.

    Concrete subclasses must implement ``apply`` (forward ``A``) and
    ``adjoint`` (``A^T`` for linear ops, Jacobian-vector-product for
    non-linear ops). ``pseudoinverse`` defaults to the adjoint; override
    when an analytical inverse is known (e.g. FBP for Radon, filtered
    inverse DFT for fully-sampled Fourier).

    The operator is stateful: it carries whatever geometry / mask / kernel
    it needs at construction. An env instantiates one ``ForwardOperator``
    per problem instance (or per cohort of instances sharing geometry).
    """

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Forward model: ``x`` (ground truth) → ``y`` (measurement)."""

    @abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint ``A^T`` for linear ops, or JVP for nonlinear ops."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging / metadata."""

    def pseudoinverse(self, y: np.ndarray) -> np.ndarray:
        """Default pseudo-inverse via adjoint. Override for analytical inverses."""
        return self.adjoint(y)

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"<{self.__class__.__name__} name={self.name!r}>"
