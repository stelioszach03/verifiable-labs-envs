"""Verifiable Labs env wrapper: code-humaneval.

Single-turn procedural code-execution RL environment with sandboxed
pytest scoring and conformal coverage.

Thin re-export over ``verifiable_labs_envs.envs``; the monorepo is
the source of truth. This package exists so the env can be installed
and discovered independently via the verifiers / Prime Intellect Hub
entry-point mechanism.
"""
from verifiable_labs_envs.envs.code_humaneval import (
    load_environment as _load_environment_base,
)

ENV_NAME = "code-humaneval"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``code-humaneval`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
