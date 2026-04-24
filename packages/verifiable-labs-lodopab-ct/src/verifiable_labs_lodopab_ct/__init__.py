"""Verifiable Labs env wrapper: lodopab-ct-simplified.\n\n2D parallel-beam CT (phantom or real LoDoPaB-CT slices via use_real_data)\n\nThin re-export over ``verifiable_labs_envs.envs``; the monorepo is the source of truth. This package exists so the env can be installed and discovered independently via the verifiers / Prime Intellect Hub entry-point mechanism."""
from verifiable_labs_envs.envs.lodopab_ct import load_environment as _load_environment_base

ENV_NAME = "lodopab-ct-simplified"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``lodopab-ct-simplified`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
