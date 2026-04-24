"""LLM adapter for the primitive-composition sparse-Fourier tool-use env."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "sparse-fourier-recovery-tools"

SYSTEM_PROMPT_TOOLS = """You are an expert at sparse signal recovery (compressed sensing).

You have access to 5 Python primitive tools. NONE of them returns a full
reconstruction on its own — each performs a single linear-algebra or
proximal operation. You must compose them yourself, over multiple iterations,
to recover the sparse signal.

Available primitives (all operate on integer x1000 fixed-point values):

- fft_tool(signal_x1000):
    Forward operator A = S·F. Input is a length-n dense real signal.
    Returns m complex Fourier coefficients at the observed mask positions.

- ifft_tool(spectrum_re_x1000, spectrum_im_x1000):
    Adjoint of A (zero-fill at the mask + inverse DFT). Input is m complex
    coefficients. Returns a length-n dense real signal.

- threshold_tool(signal_x1000, tau_x1000):
    Elementwise soft-threshold (the ISTA proximal step):
        sign(x) * max(|x| - tau, 0)
    Shrinks every entry by tau; entries smaller than tau become exactly 0.

- compute_residual_tool(signal_x1000):
    Given a dense candidate x, returns r = y - A(x) (the measurement
    residual) plus its L2 and max-abs norms. Use this to monitor convergence.

- sparsity_norm_tool(signal_x1000):
    Returns ||x||_1, ||x||_2 and the count of entries with |x| > 1e-3.
    Useful when tuning tau.

High-level recipe (you choose step size eta and threshold tau yourself):

    x = ifft(y)                          # dirty-image initialization
    for _ in range(~10-20):
        r = compute_residual(x)          # measurement residual
        g = ifft(r.real, r.imag)         # pre-conditioned gradient direction
        x = threshold(x + eta*g, tau)    # proximal update, shrinks small entries

The residual L2 should monotonically decrease. Stop when it plateaus or when
the nonzero-count of x matches the known k (given in the user prompt).

You may emit up to 30 tool calls per episode. After them (or earlier, when
you are confident), output a final JSON object with keys "support_idx"
(k sorted integers in [0, n)) and "support_amp_x1000" (k integers, same
order). No prose, no markdown fences around the final answer."""


class SparseFourierToolsAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_TOOLS

    def __init__(self) -> None:
        self._base = SparseFourierLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)


__all__ = ["SparseFourierToolsAdapter", "NAME", "SYSTEM_PROMPT_TOOLS"]
