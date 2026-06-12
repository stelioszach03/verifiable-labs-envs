"""Verifiable Labs env wrapper: long-context-needle.

Single-turn long-context needle-in-haystack RL environment with a
procedurally generated multi-document corpus, position-varied
needle injection, and conformal coverage. Thin re-export over
``verifiable_labs_envs.envs``; the monorepo is the source of truth.
"""
from verifiable_labs_envs.envs.long_context_needle import (
    load_environment as _load_environment_base,
)

ENV_NAME = "long-context-needle"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``long-context-needle`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
