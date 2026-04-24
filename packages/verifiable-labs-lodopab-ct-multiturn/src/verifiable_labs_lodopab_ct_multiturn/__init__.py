"""Verifiable Labs env wrapper: lodopab-ct-simplified-multiturn.\n\n3-turn CT reconstruction with FBP-domain residual feedback\n\nThin re-export over ``verifiable_labs_envs.envs``; the monorepo is the source of truth. This package exists so the env can be installed and discovered independently via the verifiers / Prime Intellect Hub entry-point mechanism."""
from verifiable_labs_envs.envs.lodopab_ct_multiturn import load_environment as _load_environment_base

ENV_NAME = "lodopab-ct-simplified-multiturn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``lodopab-ct-simplified-multiturn`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
