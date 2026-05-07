"""Verifiable Labs env wrapper: math-algebra.

Single-turn algebraic-simplification RL environment with
SymPy-verified rewards and conformal coverage.

Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is the
source of truth. This package exists so the env can be installed and
discovered independently via the verifiers / Prime Intellect Hub
entry-point mechanism."""
from verifiable_labs_envs.envs.math_algebra import load_environment as _load_environment_base

ENV_NAME = "math-algebra"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``math-algebra`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
