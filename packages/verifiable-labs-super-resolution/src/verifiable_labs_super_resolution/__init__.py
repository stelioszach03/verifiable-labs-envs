"""Verifiable Labs env wrapper: super-resolution-div2k-x4.\n\n4× single-image super-resolution with bicubic baseline\n\nThin re-export over ``verifiable_labs_envs.envs``; the monorepo is the source of truth. This package exists so the env can be installed and discovered independently via the verifiers / Prime Intellect Hub entry-point mechanism."""
from verifiable_labs_envs.envs.super_resolution import load_environment as _load_environment_base

ENV_NAME = "super-resolution-div2k-x4"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``super-resolution-div2k-x4`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
