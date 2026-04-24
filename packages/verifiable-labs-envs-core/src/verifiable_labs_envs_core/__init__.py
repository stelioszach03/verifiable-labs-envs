"""Verifiable Labs core — shared utilities re-exported from the monorepo.

This is a thin wrapper around the source-of-truth ``verifiable_labs_envs``
package. Installing ``verifiable-labs-envs-core`` pulls in the full repo
and exposes the shared modules:
- ``conformal``        — split-conformal calibration + coverage score
- ``forward_ops``      — shared forward operators (Fourier, blur, Radon)
- ``solvers``          — LLMSolver base + OpenRouterSolver + FakeLLMSolver
- ``solvers.adapters`` — per-env prompt/parser adapters

Use this package as a dependency when building downstream env packages
that want a stable API for the shared primitives.
"""
from verifiable_labs_envs import conformal, forward_ops
from verifiable_labs_envs.solvers import (
    CompletionResult,
    EnvAdapter,
    FakeLLMSolver,
    HAS_OPENROUTER_KEY,
    LLMSolver,
    LLMSolverError,
    OpenRouterSolver,
    register_adapter,
)

__all__ = [
    "conformal",
    "forward_ops",
    "CompletionResult",
    "EnvAdapter",
    "FakeLLMSolver",
    "HAS_OPENROUTER_KEY",
    "LLMSolver",
    "LLMSolverError",
    "OpenRouterSolver",
    "register_adapter",
]
