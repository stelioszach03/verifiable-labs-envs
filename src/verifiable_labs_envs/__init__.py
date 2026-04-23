"""verifiable-labs-envs: RL environments for scientific reasoning."""

__version__ = "0.0.1"

_REGISTRY: dict[str, str] = {
    "sparse-fourier-recovery": "verifiable_labs_envs.envs.sparse_fourier",
    "super-resolution-div2k-x4": "verifiable_labs_envs.envs.super_resolution",
    "lodopab-ct-simplified": "verifiable_labs_envs.envs.lodopab_ct",
}


def load_environment(name: str):
    """Load an environment by registered name.

    Mirrors the ``verifiers.load_environment`` signature for forward
    compatibility with the Prime Intellect Environments Hub, but returns
    our local ``Environment`` object which supports array-valued I/O.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    import importlib

    module = importlib.import_module(_REGISTRY[name])
    return module.load_environment()


def list_environments() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["__version__", "load_environment", "list_environments"]
