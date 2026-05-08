"""Verifiable Labs env wrapper: sql-single-turn.

Single-turn text-to-SQL RL environment with SQLite sandbox,
result-set comparator, and conformal coverage. Thin re-export over
``verifiable_labs_envs.envs``; the monorepo is the source of truth.
"""
from verifiable_labs_envs.envs.sql_single_turn import (
    load_environment as _load_environment_base,
)

ENV_NAME = "sql-single-turn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``sql-single-turn`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
