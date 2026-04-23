"""LLM adapter for the multi-turn sparse-Fourier environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.forward_ops import sparse_fourier_forward
from verifiable_labs_envs.solvers.adapters.sparse_fourier import SparseFourierLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "sparse-fourier-recovery-multiturn"

SYSTEM_PROMPT_MT = """You are an expert at sparse signal recovery (compressed sensing).

You have up to 3 turns. On turn 1 you see the full problem (n, k, sigma, mask,
y). Propose k non-zero positions and their real amplitudes x1000.

On turns 2 and 3 you see FEEDBACK: the Fourier-domain residual r = y - A(x_hat)
of your previous answer (scaled x1000, real and imaginary parts). The closer
this residual is to zero, the better your previous answer was. Use it to
propose a CORRECTED (support_idx, support_amp_x1000) with the same schema.

Always output exactly one JSON object with keys "support_idx" (k sorted
integers in [0, n)) and "support_amp_x1000" (k signed integers, same order).
No prose, no markdown fences, no explanations."""


class SparseFourierMultiturnAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_MT

    def __init__(self) -> None:
        self._base = SparseFourierLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],  # noqa: ARG002
        last_prediction: Any,
        instance: Any,
    ) -> str:
        x_hat_c = last_prediction.x_hat.astype(np.complex128)
        y_hat = sparse_fourier_forward(x_hat_c, instance.mask)
        residual = np.asarray(instance.y) - np.asarray(y_hat)
        residual_re = [int(round(v * 1000)) for v in residual.real.tolist()]
        residual_im = [int(round(v * 1000)) for v in residual.imag.tolist()]
        payload = {
            "residual_re_x1000": residual_re,
            "residual_im_x1000": residual_im,
        }
        return (
            "FEEDBACK: residual r = y - A(x_hat) of your previous answer "
            "(scaled x1000, aligned with the original mask order):\n"
            + json.dumps(payload, separators=(",", ":"))
            + "\n\nPropose a CORRECTED (support_idx, support_amp_x1000) using the same schema."
        )


__all__ = ["SparseFourierMultiturnAdapter", "NAME", "SYSTEM_PROMPT_MT"]
