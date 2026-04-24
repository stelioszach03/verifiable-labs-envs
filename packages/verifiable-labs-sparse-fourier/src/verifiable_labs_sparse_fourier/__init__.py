"""Verifiable Labs env wrapper: sparse-fourier-recovery.\n\n1D sparse Fourier recovery with OMP baseline + conformal σ̂\n\nThin re-export over ``verifiable_labs_envs.envs``; the monorepo is the source of truth. This package exists so the env can be installed and discovered independently via the verifiers / Prime Intellect Hub entry-point mechanism."""
from verifiable_labs_envs.envs.sparse_fourier import load_environment as _load_environment_base

ENV_NAME = "sparse-fourier-recovery"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``sparse-fourier-recovery`` environment (delegates to the monorepo)."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
