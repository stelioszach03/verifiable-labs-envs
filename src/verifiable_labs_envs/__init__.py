"""verifiable-labs-envs: RL environments for scientific reasoning."""

__version__ = "0.0.1"

_REGISTRY: dict[str, str] = {
    "sparse-fourier-recovery": "verifiable_labs_envs.envs.sparse_fourier",
    "sparse-fourier-recovery-multiturn": "verifiable_labs_envs.envs.sparse_fourier_multiturn",
    "sparse-fourier-recovery-tools": "verifiable_labs_envs.envs.sparse_fourier_tools",
    "super-resolution-div2k-x4": "verifiable_labs_envs.envs.super_resolution",
    "lodopab-ct-simplified": "verifiable_labs_envs.envs.lodopab_ct",
    "lodopab-ct-simplified-multiturn": "verifiable_labs_envs.envs.lodopab_ct_multiturn",
    "phase-retrieval": "verifiable_labs_envs.envs.phase_retrieval",
    "phase-retrieval-multiturn": "verifiable_labs_envs.envs.phase_retrieval_multiturn",
}


def load_environment(name: str, **kwargs):
    """Load an environment by registered name.

    Mirrors the ``verifiers.load_environment`` signature for forward
    compatibility with the Prime Intellect Environments Hub. ``kwargs``
    are forwarded to the per-env ``load_environment`` factory (e.g.
    ``calibration_quantile=...`` or ``use_real_data=True``).
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    import importlib

    module = importlib.import_module(_REGISTRY[name])
    return module.load_environment(**kwargs)


def list_environments() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["__version__", "load_environment", "list_environments"]
