"""Verifiable Labs env wrapper: sql-multiturn.

Multi-turn text-to-SQL RL environment with verifier-feedback
rollouts and per-extra-turn penalty. Thin re-export over
``verifiable_labs_envs.envs``.
"""
from verifiable_labs_envs.envs.sql_multiturn import (
    load_environment as _load_environment_base,
)

ENV_NAME = "sql-multiturn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``sql-multiturn`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
