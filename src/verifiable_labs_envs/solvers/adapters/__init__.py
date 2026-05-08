"""Per-environment LLM adapters. Importing this package registers all of them."""
from verifiable_labs_envs.solvers.adapters.code_humaneval import (
    CodeHumanevalLLMAdapter,
)
from verifiable_labs_envs.solvers.adapters.code_humaneval_multiturn import (
    CodeHumanevalMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.code_humaneval_tools import (
    CodeHumanevalToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.code_mini_repo import (
    CodeMiniRepoAdapter,
)
from verifiable_labs_envs.solvers.adapters.lodopab_ct import LodopabCtLLMAdapter
from verifiable_labs_envs.solvers.adapters.lodopab_ct_multiturn import (
    LodopabCtMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.math_algebra import MathAlgebraLLMAdapter
from verifiable_labs_envs.solvers.adapters.math_algebra_multiturn import (
    MathAlgebraMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.math_algebra_tools import (
    MathAlgebraToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.mri_knee import MRIKneeLLMAdapter
from verifiable_labs_envs.solvers.adapters.mri_knee_multiturn import (
    MRIKneeMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.phase_retrieval import (
    PhaseRetrievalLLMAdapter,
)
from verifiable_labs_envs.solvers.adapters.phase_retrieval_multiturn import (
    PhaseRetrievalMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.adapters.sparse_fourier_multiturn import (
    SparseFourierMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.sparse_fourier_tools import (
    SparseFourierToolsAdapter,
)
from verifiable_labs_envs.solvers.adapters.super_resolution import SuperResolutionLLMAdapter
from verifiable_labs_envs.solvers.adapters.tool_calling_debug import (
    ToolCallingDebugAdapter,
)
from verifiable_labs_envs.solvers.adapters.tool_calling_multiturn import (
    ToolCallingMultiturnAdapter,
)
from verifiable_labs_envs.solvers.adapters.tool_calling_single import (
    ToolCallingSingleAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import register_adapter

register_adapter(SparseFourierLLMAdapter())
register_adapter(SparseFourierMultiturnAdapter())
register_adapter(SparseFourierToolsAdapter())
register_adapter(SuperResolutionLLMAdapter())
register_adapter(LodopabCtLLMAdapter())
register_adapter(LodopabCtMultiturnAdapter())
register_adapter(PhaseRetrievalLLMAdapter())
register_adapter(PhaseRetrievalMultiturnAdapter())
register_adapter(MRIKneeLLMAdapter())
register_adapter(MRIKneeMultiturnAdapter())
register_adapter(MathAlgebraLLMAdapter())
register_adapter(MathAlgebraMultiturnAdapter())
register_adapter(MathAlgebraToolsAdapter())
# Phase 24 — code-execution env family.
register_adapter(CodeHumanevalLLMAdapter())
register_adapter(CodeHumanevalMultiturnAdapter())
register_adapter(CodeHumanevalToolsAdapter())
register_adapter(CodeMiniRepoAdapter())
# Phase 25 — tool-calling env family.
register_adapter(ToolCallingSingleAdapter())
register_adapter(ToolCallingMultiturnAdapter())
register_adapter(ToolCallingDebugAdapter())

__all__ = [
    "SparseFourierLLMAdapter",
    "SparseFourierMultiturnAdapter",
    "SparseFourierToolsAdapter",
    "SuperResolutionLLMAdapter",
    "LodopabCtLLMAdapter",
    "LodopabCtMultiturnAdapter",
    "PhaseRetrievalLLMAdapter",
    "PhaseRetrievalMultiturnAdapter",
    "MRIKneeLLMAdapter",
    "MRIKneeMultiturnAdapter",
    "MathAlgebraLLMAdapter",
    "MathAlgebraMultiturnAdapter",
    "MathAlgebraToolsAdapter",
    "CodeHumanevalLLMAdapter",
    "CodeHumanevalMultiturnAdapter",
    "CodeHumanevalToolsAdapter",
    "CodeMiniRepoAdapter",
    "ToolCallingSingleAdapter",
    "ToolCallingMultiturnAdapter",
    "ToolCallingDebugAdapter",
]
