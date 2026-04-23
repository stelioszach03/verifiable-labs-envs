"""Per-environment LLM adapters. Importing this package registers all three."""
from verifiable_labs_envs.solvers.adapters.lodopab_ct import LodopabCtLLMAdapter
from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.adapters.super_resolution import SuperResolutionLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import register_adapter

register_adapter(SparseFourierLLMAdapter())
register_adapter(SuperResolutionLLMAdapter())
register_adapter(LodopabCtLLMAdapter())

__all__ = [
    "SparseFourierLLMAdapter",
    "SuperResolutionLLMAdapter",
    "LodopabCtLLMAdapter",
]
