"""Per-environment LLM adapters. Importing this package registers all of them."""
from verifiable_labs_envs.solvers.adapters.lodopab_ct import LodopabCtLLMAdapter
from verifiable_labs_envs.solvers.adapters.lodopab_ct_multiturn import (
    LodopabCtMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.adapters.sparse_fourier_multiturn import (
    SparseFourierMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.sparse_fourier_tools import (
    SparseFourierToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.super_resolution import SuperResolutionLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import register_adapter

register_adapter(SparseFourierLLMAdapter())
register_adapter(SparseFourierMultiturnAdapter())
register_adapter(SparseFourierToolsAdapter())
register_adapter(SuperResolutionLLMAdapter())
register_adapter(LodopabCtLLMAdapter())
register_adapter(LodopabCtMultiturnAdapter())

__all__ = [
    "SparseFourierLLMAdapter",
    "SparseFourierMultiturnAdapter",
    "SparseFourierToolsAdapter",
    "SuperResolutionLLMAdapter",
    "LodopabCtLLMAdapter",
    "LodopabCtMultiturnAdapter",
]
