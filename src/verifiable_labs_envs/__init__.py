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
    "mri-knee-reconstruction": "verifiable_labs_envs.envs.mri_knee",
    "mri-knee-reconstruction-multiturn": "verifiable_labs_envs.envs.mri_knee_multiturn",
    "math-algebra": "verifiable_labs_envs.envs.math_algebra",
    "math-algebra-multiturn": "verifiable_labs_envs.envs.math_algebra_multiturn",
    "math-algebra-tools": "verifiable_labs_envs.envs.math_algebra_tools",
    # Phase 24 — code-execution env family.
    "code-humaneval": "verifiable_labs_envs.envs.code_humaneval",
    "code-humaneval-multiturn": "verifiable_labs_envs.envs.code_humaneval_multiturn",
    "code-humaneval-tools": "verifiable_labs_envs.envs.code_humaneval_tools",
    "code-mini-repo": "verifiable_labs_envs.envs.code_mini_repo",
    # Phase 25 — tool-calling env family.
    "tool-calling-single": "verifiable_labs_envs.envs.tool_calling_single",
    "tool-calling-multiturn": "verifiable_labs_envs.envs.tool_calling_multiturn",
    "tool-calling-debug": "verifiable_labs_envs.envs.tool_calling_debug",
    # Phase 26 — sql env family.
    "sql-single-turn": "verifiable_labs_envs.envs.sql_single_turn",
    "sql-multiturn": "verifiable_labs_envs.envs.sql_multiturn",
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
