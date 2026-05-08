"""Verifiable Labs env wrapper: tool-calling-debug.

Trace-debug procedural tool-calling RL environment with
prefix-conditioned trajectory completion. Thin re-export over
``verifiable_labs_envs.envs``.
"""
from verifiable_labs_envs.envs.tool_calling_debug import (
    load_environment as _load_environment_base,
)

ENV_NAME = "tool-calling-debug"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``tool-calling-debug`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
