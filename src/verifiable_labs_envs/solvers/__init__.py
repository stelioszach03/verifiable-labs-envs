"""LLM-based solvers and per-environment prompt / response adapters."""
# Importing `adapters` registers the three per-env adapters on import.
from verifiable_labs_envs.solvers import adapters as _adapters  # noqa: F401
from verifiable_labs_envs.solvers.llm_solver import (
    HAS_OPENROUTER_KEY,
    CompletionResult,
    EnvAdapter,
    FakeLLMSolver,
    LLMSolver,
    LLMSolverError,
    OpenRouterSolver,
    register_adapter,
)

__all__ = [
    "HAS_OPENROUTER_KEY",
    "CompletionResult",
    "EnvAdapter",
    "FakeLLMSolver",
    "LLMSolver",
    "LLMSolverError",
    "OpenRouterSolver",
    "register_adapter",
]
