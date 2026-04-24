"""LLM adapter for the tool-use sparse-Fourier environment."""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "sparse-fourier-recovery-tools"

SYSTEM_PROMPT_TOOLS = """You are an expert at sparse signal recovery (compressed sensing).

You have access to 4 Python tools you can call strategically before committing
to your final answer:

- fft_tool(support_idx, support_amp_x1000): apply A=S.F to a candidate signal.
- ifft_tool(real_x1000_at_mask, imag_x1000_at_mask): inverse-DFT of zero-filled
  measurements (the "dirty image"; a natural first-guess heuristic).
- ista_tool(): run the classical OMP solver on THIS instance; returns a strong
  baseline answer you can critique or refine.
- check_residual_tool(support_idx, support_amp_x1000): compute ||y - A(x_hat)||_2
  for a candidate without committing.

You may emit up to 5 tool calls per episode. After them, output a final JSON
object with keys "support_idx" (k sorted integers in [0, n)) and
"support_amp_x1000" (k integers, same order). No prose, no markdown fences
around the final answer."""


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
