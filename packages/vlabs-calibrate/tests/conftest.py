"""Shared pytest fixtures for vlabs-calibrate tests."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests; seed is fixed."""
    return np.random.default_rng(42)


@pytest.fixture
def gaussian_traces() -> Callable[[int, int, float], list[dict]]:
    """Factory: ``gaussian_traces(n, seed, sigma=0.5)`` returns ``n`` traces.

    Each trace has the shape::

        {"x": float, "reference_reward": float, "uncertainty": float}

    where ``reference_reward = x + sigma * N(0, 1)`` so the calibration
    target with ``reward_fn = lambda x: x`` is well-behaved.
    """

    def _make(n: int, seed: int, sigma: float = 0.5) -> list[dict]:
        gen = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            x = float(gen.standard_normal())
            ref = x + sigma * float(gen.standard_normal())
            out.append({"x": x, "reference_reward": ref, "uncertainty": sigma})
        return out

    return _make
