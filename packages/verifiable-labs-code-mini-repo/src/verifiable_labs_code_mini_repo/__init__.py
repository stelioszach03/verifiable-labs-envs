"""Verifiable Labs env wrapper: code-mini-repo.

Synthetic-mini-repo code-execution RL environment with multi-file
edit and conformal coverage. Thin re-export over
``verifiable_labs_envs.envs``.
"""
from verifiable_labs_envs.envs.code_mini_repo import (
    load_environment as _load_environment_base,
)

ENV_NAME = "code-mini-repo"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``code-mini-repo`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
