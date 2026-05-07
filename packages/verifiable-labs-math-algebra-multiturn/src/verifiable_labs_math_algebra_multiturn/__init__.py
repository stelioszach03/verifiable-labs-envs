"""Verifiable Labs env wrapper: math-algebra-multiturn.

Multi-turn algebraic-simplification RL environment with verifier
feedback and SymPy-verified rewards.

Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is the
source of truth."""
from verifiable_labs_envs.envs.math_algebra_multiturn import (
    load_environment as _load_environment_base,
)

ENV_NAME = "math-algebra-multiturn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``math-algebra-multiturn`` environment (delegates to monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
