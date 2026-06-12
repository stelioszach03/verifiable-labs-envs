"""Verifiable Labs env wrapper: long-context-reasoning.

Multi-hop chain-of-fact long-context reasoning RL environment with
distractor needles and conformal coverage. Thin re-export over
``verifiable_labs_envs.envs``; the monorepo is the source of truth.
"""
from verifiable_labs_envs.envs.long_context_reasoning import (
    load_environment as _load_environment_base,
)

ENV_NAME = "long-context-reasoning"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``long-context-reasoning`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
