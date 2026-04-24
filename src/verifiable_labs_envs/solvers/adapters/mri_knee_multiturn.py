"""LLM adapter for the multi-turn MRI-knee-reconstruction env."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.mri_knee import Instance, Prediction
from verifiable_labs_envs.forward_ops import FFTMask2DOp
from verifiable_labs_envs.solvers.adapters.mri_knee import MRIKneeLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "mri-knee-reconstruction-multiturn"

SYSTEM_PROMPT_MULTITURN = """You are an expert MRI reconstruction solver operating over up to 3 turns.

Turn 1: I send the zero-filled-IFFT image and ask for a refined image.
Turn 2-3: I send the k-space residual (y - M·F(x_hat_prev)) summary so you can
correct the unmodeled aliasing pattern. Each turn you emit a full image.

Final answer format, every turn: {"image": [[int, int, ...], ...]} with pixels
in [0, 255] at the given resolution. No prose, no markdown fences."""


class MRIKneeMultiturnAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_MULTITURN

    def __init__(self) -> None:
        self._base = MRIKneeLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)

    def build_followup_turn(
        self, history: list[dict[str, str]], prediction: Prediction, instance: Instance
    ) -> str:  # noqa: ARG002
        op = FFTMask2DOp(instance.mask)
        y_hat = op.apply(prediction.x_hat)
        residual_complex = instance.y - y_hat  # shape=(h,w)
        # Compress residual into a small summary: norms + a compact k-space
        # residual-magnitude image at the same resolution.
        res_mag = np.abs(residual_complex)
        l2 = float(np.linalg.norm(residual_complex))
        max_abs = float(np.max(res_mag)) if res_mag.size else 0.0
        # Rescale residual magnitude to int in [0, 255] for compact JSON.
        peak = max(max_abs, 1e-9)
        res_u8 = (res_mag / peak * 255.0).round().astype(np.int64)

        payload = {
            "residual_l2_x1000": int(round(l2 * 1000)),
            "residual_max_abs_x1000": int(round(max_abs * 1000)),
            "residual_mag_u8_peak_normalized": [
                [int(v) for v in row] for row in res_u8.tolist()
            ],
        }
        return (
            "K-SPACE RESIDUAL from your previous answer (y − M·F(x_hat_prev)):\n"
            + json.dumps(payload, separators=(",", ":"))
            + "\n\nRevise the image and output the JSON object {\"image\": ...}."
        )


__all__ = ["MRIKneeMultiturnAdapter", "SYSTEM_PROMPT_MULTITURN"]
