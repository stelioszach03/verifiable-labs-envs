"""Env registry helpers exposed to the API layer.

Wraps ``verifiable_labs_envs.list_environments`` with API-shaped
metadata (domain, multi-turn / tool-use flags) so ``GET /v1/environments``
can return a useful payload without re-introspecting the env code on
every request.
"""
from __future__ import annotations

from dataclasses import dataclass

from verifiable_labs_envs import list_environments


@dataclass(frozen=True)
class EnvMeta:
    id: str
    qualified_id: str  # "stelioszach/<id>" — matches Prime Hub naming
    domain: str
    multi_turn: bool
    tool_use: bool
    description: str


_DOMAIN: dict[str, str] = {
    "sparse-fourier-recovery": "compressed-sensing",
    "sparse-fourier-recovery-multiturn": "compressed-sensing",
    "sparse-fourier-recovery-tools": "compressed-sensing",
    "super-resolution-div2k-x4": "image-super-resolution",
    "lodopab-ct-simplified": "medical-imaging-ct",
    "lodopab-ct-simplified-multiturn": "medical-imaging-ct",
    "phase-retrieval": "coherent-diffraction",
    "phase-retrieval-multiturn": "coherent-diffraction",
    "mri-knee-reconstruction": "medical-imaging-mri",
    "mri-knee-reconstruction-multiturn": "medical-imaging-mri",
    "math-algebra": "symbolic-algebra",
    "math-algebra-multiturn": "symbolic-algebra",
    "math-algebra-tools": "symbolic-algebra",
    # Phase 24 — code-execution env family.
    "code-humaneval": "code-execution",
    "code-humaneval-multiturn": "code-execution",
    "code-humaneval-tools": "code-execution",
    "code-mini-repo": "code-execution",
    # Phase 25 — tool-calling env family.
    "tool-calling-single": "tool-calling",
    "tool-calling-multiturn": "tool-calling",
    "tool-calling-debug": "tool-calling",
}


_DESCRIPTIONS: dict[str, str] = {
    "sparse-fourier-recovery":
        "1D sparse Fourier recovery from m of n DFT coefficients with OMP "
        "baseline and conformal sigma-hat.",
    "sparse-fourier-recovery-multiturn":
        "Three-turn dialogue variant of sparse Fourier recovery; server "
        "feeds the Fourier-domain residual back between turns.",
    "sparse-fourier-recovery-tools":
        "Primitive-composition tool-use sparse-Fourier env (v0.3): fft, "
        "ifft, soft-threshold, compute_residual, sparsity_norm.",
    "super-resolution-div2k-x4":
        "Single-image 4x super-resolution on DIV2K-style ground truth "
        "with bicubic baseline and edge-weighted sigma-hat.",
    "lodopab-ct-simplified":
        "2D parallel-beam CT reconstruction (phantom or real LoDoPaB "
        "slices) with FBP baseline.",
    "lodopab-ct-simplified-multiturn":
        "Three-turn CT reconstruction with FBP-domain residual feedback.",
    "phase-retrieval":
        "Sparse phase retrieval from magnitude-only DFT coefficients with "
        "Gerchberg-Saxton baseline. Sign-invariant scoring.",
    "phase-retrieval-multiturn":
        "Three-turn phase retrieval with magnitude-residual feedback "
        "between turns.",
    "mri-knee-reconstruction":
        "Accelerated MRI from 4x-undersampled Cartesian k-space with "
        "zero-filled inverse-FFT baseline.",
    "mri-knee-reconstruction-multiturn":
        "Three-turn MRI knee reconstruction with k-space residual "
        "feedback.",
    "math-algebra":
        "Single-turn symbolic-algebra simplification (expand, factor, "
        "collect-like-terms) with SymPy-verified rewards and conformal "
        "coverage; threaded simplify timeout.",
    "math-algebra-multiturn":
        "Three-turn symbolic-algebra dialogue with verifier feedback at "
        "the format / parse / equivalence layer; gold expression never "
        "revealed to the solver.",
    "math-algebra-tools":
        "Tool-use symbolic-algebra env exposing sympy_simplify, "
        "sympy_expand, sympy_solve, sympy_substitute primitives; "
        "model composes them to reach the answer.",
    "code-humaneval":
        "Single-turn procedural code-execution env across 12 templates "
        "(lists, strings, dicts, ints, trees, graphs); sandboxed pytest "
        "scoring under D5 rlimits and `unshare -rn` network isolation.",
    "code-humaneval-multiturn":
        "Three-turn dialogue variant of code-humaneval; visible-test "
        "pass counts feed back between turns, hidden tests held out.",
    "code-humaneval-tools":
        "Tool-use code-execution env exposing read_file, write_file, "
        "run_test primitives; model iterates on a per-call workspace.",
    "code-mini-repo":
        "Repo-scale synthetic code-execution env: 3-file mini-repos "
        "(bug-fix, feature-add, refactor-preserve) with multi-file edits.",
    "tool-calling-single":
        "Single-pass procedural tool-calling env across 10 templates "
        "(arithmetic, search-and-email, file-concat, multi-search, "
        "outbox-audit) with mock primitives and D2-C composite reward.",
    "tool-calling-multiturn":
        "Multi-turn tool-calling env with verifier-feedback rollouts "
        "and per-extra-turn penalty (cap 10%); same template lattice "
        "as tool-calling-single.",
    "tool-calling-debug":
        "Trace-debug tool-calling env: each instance ships a partial "
        "tool-call trajectory plus the workspace snapshot it produced; "
        "the model continues from that point.",
}


def _qualify(env_id: str) -> str:
    return f"stelioszach/{env_id}"


def all_envs() -> list[EnvMeta]:
    """Return metadata for every registered env, sorted by id."""
    metas: list[EnvMeta] = []
    for env_id in list_environments():
        metas.append(EnvMeta(
            id=env_id,
            qualified_id=_qualify(env_id),
            domain=_DOMAIN.get(env_id, "unknown"),
            multi_turn=env_id.endswith("-multiturn"),
            tool_use=env_id.endswith("-tools"),
            description=_DESCRIPTIONS.get(
                env_id, "Verifiable Labs environment."
            ),
        ))
    return metas


def normalize_env_id(env_id: str) -> str:
    """Strip ``owner/`` prefix; the env registry uses bare ids."""
    if "/" in env_id:
        return env_id.rsplit("/", 1)[-1]
    return env_id
