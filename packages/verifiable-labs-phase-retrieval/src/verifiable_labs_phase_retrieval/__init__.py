"""Verifiable Labs env wrapper: phase-retrieval.

Phase retrieval from magnitude-only Fourier measurements with Gerchberg-Saxton
baseline + conformal σ̂. Thin re-export over ``verifiable_labs_envs.envs``; the
monorepo at https://github.com/stelioszach03/verifiable-labs-envs is the source
of truth.
"""
from verifiable_labs_envs.envs.phase_retrieval import (
    load_environment as _load_environment_base,
)
from verifiable_labs_envs.envs.phase_retrieval_multiturn import (
    load_environment as _load_environment_multiturn_base,
)

ENV_NAME = "phase-retrieval"
MT_ENV_NAME = "phase-retrieval-multiturn"
__version__ = "1.0.0"


def load_environment(*args, **kwargs):
    """Factory for the single-turn ``phase-retrieval`` environment."""
    return _load_environment_base(*args, **kwargs)


def load_environment_multiturn(*args, **kwargs):
    """Factory for the 3-turn ``phase-retrieval-multiturn`` environment."""
    return _load_environment_multiturn_base(*args, **kwargs)


__all__ = [
    "ENV_NAME",
    "MT_ENV_NAME",
    "load_environment",
    "load_environment_multiturn",
    "__version__",
]
