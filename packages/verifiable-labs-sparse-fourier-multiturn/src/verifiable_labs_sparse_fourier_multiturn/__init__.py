"""Verifiable Labs env wrapper: sparse-fourier-recovery-multiturn.\n\n3-turn sparse Fourier recovery with residual feedback between turns\n\nThin re-export over ``verifiable_labs_envs.envs``; the monorepo is the source of truth. This package exists so the env can be installed and discovered independently via the verifiers / Prime Intellect Hub entry-point mechanism."""
from verifiable_labs_envs.envs.sparse_fourier_multiturn import load_environment as _load_environment_base

ENV_NAME = "sparse-fourier-recovery-multiturn"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``sparse-fourier-recovery-multiturn`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
