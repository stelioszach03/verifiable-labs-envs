"""Verifiable Labs env wrapper: tool-calling-single.

Single-pass procedural tool-calling RL environment with mock
primitives + D2-C composite reward + conformal coverage. Thin
re-export over ``verifiable_labs_envs.envs``; the monorepo is the
source of truth.
"""
from verifiable_labs_envs.envs.tool_calling_single import (
    load_environment as _load_environment_base,
)

ENV_NAME = "tool-calling-single"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``tool-calling-single`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
