"""Verifiable Labs env wrapper: long-context-synthesis.

Multi-needle 3-turn long-context synthesis RL environment with
token-F1 scoring and conformal coverage. Thin re-export over
``verifiable_labs_envs.envs``; the monorepo is the source of truth.
"""
from verifiable_labs_envs.envs.long_context_synthesis import (
    load_environment as _load_environment_base,
)

ENV_NAME = "long-context-synthesis"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``long-context-synthesis`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
