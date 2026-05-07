"""Verifiable Labs env wrapper: math-algebra-tools.

Tool-use algebraic-simplification RL environment with SymPy primitive
tools and SymPy-verified rewards.

Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is the
source of truth."""
from verifiable_labs_envs.envs.math_algebra_tools import (
    load_environment as _load_environment_base,
)

ENV_NAME = "math-algebra-tools"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``math-algebra-tools`` environment (delegates to monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
