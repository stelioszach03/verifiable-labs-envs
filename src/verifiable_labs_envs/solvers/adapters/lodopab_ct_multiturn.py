"""LLM adapter for the multi-turn lodopab-ct-simplified environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skimage import transform as sktransform

from verifiable_labs_envs.forward_ops import radon_fbp, radon_forward
from verifiable_labs_envs.solvers.adapters.lodopab_ct import (
    COARSE_SIZE,
    LodopabCtLLMAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "lodopab-ct-simplified-multiturn"

SYSTEM_PROMPT_MT = f"""You are an expert CT image-reconstruction solver.

You have up to 3 turns. On turn 1 you see a coarse 32x32 FBP reconstruction of
a 60-angle parallel-beam sinogram and propose a {COARSE_SIZE}x{COARSE_SIZE}
refined image (integer pixel values in [0, 255]).

On turns 2 and 3 you see FEEDBACK: the FBP back-projection of the sinogram
residual `r = y - R(x_hat)` from your previous answer, downsampled to
{COARSE_SIZE}x{COARSE_SIZE} and encoded as signed int8 with a scale factor.
Large-magnitude values indicate where your reconstruction disagrees with the
measurement. Use this to propose a corrected {COARSE_SIZE}x{COARSE_SIZE} image.

Always output exactly one JSON object: {{"image": [...32 rows of 32 ints
in [0, 255]...]}}. No prose, no markdown fences, no explanations."""


class LodopabCtMultiturnAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_MT

    def __init__(self) -> None:
        self._base = LodopabCtLLMAdapter()

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
        # Sinogram residual for the LLM's current reconstruction.
        sino_from_hat = radon_forward(last_prediction.x_hat, instance.angles_deg)
        sino_residual = np.asarray(instance.y) - np.asarray(sino_from_hat)
        # Back-project to image space, then down to COARSE_SIZE for the prompt.
        residual_image = radon_fbp(sino_residual, instance.angles_deg, output_size=instance.shape[0])
        residual_coarse = sktransform.resize(
            residual_image, (COARSE_SIZE, COARSE_SIZE), order=3, anti_aliasing=True, mode="reflect"
        )
        # Normalize to signed int8 with an explicit scale factor.
        abs_max = float(np.abs(residual_coarse).max())
        if abs_max > 0:
            grid = np.clip(np.round((residual_coarse / abs_max) * 127), -128, 127).astype(int)
        else:
            grid = np.zeros_like(residual_coarse, dtype=int)

        payload = {
            "residual_32x32_int8": [[int(v) for v in row] for row in grid.tolist()],
            "scale_abs_max": round(abs_max, 6),
            "note": (
                "multiply each int by (scale_abs_max / 127) to recover the real-valued residual; "
                "positive = your reconstruction is too dark there, negative = too bright."
            ),
        }
        return (
            "FEEDBACK: FBP back-projection of the sinogram residual from your previous "
            "answer, downsampled to 32x32:\n"
            + json.dumps(payload, separators=(",", ":"))
            + f"\n\nPropose a corrected {COARSE_SIZE}x{COARSE_SIZE} image with the same schema."
        )


__all__ = ["LodopabCtMultiturnAdapter", "NAME", "SYSTEM_PROMPT_MT"]
