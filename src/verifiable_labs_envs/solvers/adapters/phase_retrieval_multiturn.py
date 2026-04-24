"""LLM adapter for the multi-turn phase-retrieval environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.phase_retrieval import Instance, Prediction
from verifiable_labs_envs.forward_ops import MagnitudeOnlyOp
from verifiable_labs_envs.solvers.adapters.phase_retrieval import (
    PhaseRetrievalLLMAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "phase-retrieval-multiturn"

SYSTEM_PROMPT_MULTITURN = """You are an expert at phase retrieval. You will have up to 3 turns to refine your answer.

Turn 1: I send the problem and ask for (support_idx, support_amp_x1000).
Turn 2-3: I send the magnitude-domain residual r_mag = y - |F(x_hat_prev)| and ask for
an improved answer.

Because |F(-x)| = |F(x)|, recovery is unique only up to a global sign flip —
the scorer keeps the better of x_hat and -x_hat. Focus on recovering |x| and
the correct support first; sign is cheap.

Output format on every turn: one JSON object with keys "support_idx" (k
sorted ints in [0, n)) and "support_amp_x1000" (k ints, real amplitudes × 1000,
same order). No prose. No markdown fences."""


class PhaseRetrievalMultiturnAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_MULTITURN

    def __init__(self) -> None:
        self._base = PhaseRetrievalLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)

    def build_followup_turn(
        self, history: list[dict[str, str]], prediction: Prediction, instance: Instance
    ) -> str:  # noqa: ARG002
        """Compute magnitude residual r_mag = y - |F(x_hat)| and format it for the LLM."""
        op = MagnitudeOnlyOp(n=instance.n, mask=instance.mask)
        y_hat_mag = op.apply(prediction.x_hat)
        residual = instance.y - y_hat_mag  # can be positive or negative
        l2 = float(np.linalg.norm(residual))
        max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
        payload = {
            "residual_mag_x1000": [int(round(float(v) * 1000)) for v in residual.tolist()],
            "residual_l2_x1000": int(round(l2 * 1000)),
            "residual_max_abs_x1000": int(round(max_abs * 1000)),
        }
        return (
            "MAGNITUDE-DOMAIN RESIDUAL from your previous answer (y - |F(x_hat_prev)|):\n"
            + json.dumps(payload, separators=(", ", ": "))
            + "\n\nRevise and output the JSON object for (support_idx, support_amp_x1000)."
        )


__all__ = ["PhaseRetrievalMultiturnAdapter", "SYSTEM_PROMPT_MULTITURN"]
