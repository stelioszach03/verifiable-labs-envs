"""Verifiable Labs env wrapper: code-humaneval-multiturn.

Multi-turn procedural code-execution RL environment with
test-feedback rollouts and conformal coverage.

Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is
the source of truth.
"""
from verifiable_labs_envs.envs.code_humaneval_multiturn import (
    load_environment as _load_environment_base,
)

ENV_NAME = "code-humaneval-multiturn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``code-humaneval-multiturn`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
