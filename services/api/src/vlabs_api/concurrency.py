"""Per-env asyncio semaphores for ``POST /v1/score`` (Phase 22.C).

PHASE_22_PLAN.md §8: imaging envs (mri-knee, lodopab-ct, sparse-fourier,
phase-retrieval) are CPU-bound and dominated by FFT / Radon transforms.
A burst of imaging requests on the same Fly machine can starve the
event loop. The fix: a per-env :class:`asyncio.Semaphore` (default size
4) that limits in-flight imaging-env scoring; symbolic-math envs run
under a much larger semaphore (effectively unlimited) since each call
is sub-10 ms.

Symbolic envs explicitly listed so the default branch (``imaging``)
can stay conservative without micro-tuning per-env. New env families
(coding, SQL, …) added in later phases override their semaphore size
here.
"""
from __future__ import annotations

import asyncio
from typing import Final

# Per-env semaphore caps. Anything not listed defaults to IMAGING_SIZE.
SYMBOLIC_SIZE: Final[int] = 64
IMAGING_SIZE: Final[int] = 4

_SYMBOLIC_ENVS: Final[frozenset[str]] = frozenset(
    {
        "math-algebra",
        "math-algebra-multiturn",
        "math-algebra-tools",
    }
)

_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _semaphore_size(env_id: str) -> int:
    return SYMBOLIC_SIZE if env_id in _SYMBOLIC_ENVS else IMAGING_SIZE


def get_semaphore(env_id: str) -> asyncio.Semaphore:
    """Return the (lazily-created) semaphore for the given env."""
    sem = _SEMAPHORES.get(env_id)
    if sem is None:
        sem = asyncio.Semaphore(_semaphore_size(env_id))
        _SEMAPHORES[env_id] = sem
    return sem


def reset_for_tests() -> None:
    """Drop the global semaphore registry — only safe to call from tests."""
    _SEMAPHORES.clear()


__all__ = [
    "SYMBOLIC_SIZE",
    "IMAGING_SIZE",
    "get_semaphore",
    "reset_for_tests",
]
