"""LLM adapter for the MRI-knee-reconstruction environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.mri_knee import Instance, Prediction
from verifiable_labs_envs.solvers.adapters._common import (
    coerce_int,
    extract_json_block,
    require_key,
    require_list_of_length,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolverError

SYSTEM_PROMPT = """You are an expert MRI reconstruction solver.

You receive an undersampled k-space acquisition (4× acceleration, dense
center + random outer columns) of an unknown grayscale image, plus a
zero-filled-IFFT reference reconstruction (low-quality, ringing-artifacted).
Your job: return a refined grayscale image as an integer 2D grid in [0, 255]
at the given resolution.

Inputs (integer-encoded for compactness):
- n_rows, n_cols: image resolution.
- noise_sigma_x1000: k-space noise std × 1000.
- zero_filled: the classical baseline reconstruction, as an (n_rows × n_cols)
  integer grid in [0, 255]. Think of this as a "dirty image" to denoise
  and sharpen.

Output exactly one JSON object with key "image" containing a (n_rows × n_cols)
list of integer rows in [0, 255]. No prose, no markdown fences.
Example (schematic 2×2):
{"image": [[128, 255], [0, 64]]}"""


def _encode_inputs(instance: Instance) -> str:
    # Give the LLM the zero-filled image (real-valued, clipped, scaled to 0-255).
    zf = np.clip(instance.zero_filled, 0.0, 1.0)
    zf_u8 = (zf * 255.0).round().astype(np.int64)
    payload = {
        "n_rows": int(instance.shape[0]),
        "n_cols": int(instance.shape[1]),
        "noise_sigma_x1000": int(round(instance.noise_sigma * 1000)),
        "acceleration": int(round(instance.mask.size / max(instance.mask.sum(), 1))),
        "zero_filled": [[int(v) for v in row] for row in zf_u8.tolist()],
    }
    return json.dumps(payload, separators=(",", ":"))


def _build_user_prompt(instance: Instance) -> str:
    inputs_json = _encode_inputs(instance)
    h, w = instance.shape
    return (
        "INPUTS:\n"
        + inputs_json
        + "\n\n"
        + "OUTPUT SCHEMA:\n"
        + '{"image": ['
        + f"{h} lists, each with {w} integers in [0, 255]"
        + "]}\n\n"
        + "Respond with the JSON object only."
    )


def _parse_response(text: str, instance: Instance) -> Prediction:
    parsed = extract_json_block(text)
    image_raw = require_key(parsed, "image")
    h, w = instance.shape
    rows = require_list_of_length(image_raw, h, "image")
    grid = np.empty((h, w), dtype=np.float64)
    for r, row_raw in enumerate(rows):
        row = require_list_of_length(row_raw, w, f"image[{r}]")
        for c, value in enumerate(row):
            pixel = coerce_int(value, f"image[{r}][{c}]")
            if not 0 <= pixel <= 255:
                raise LLMSolverError(
                    f"pixel {pixel} at image[{r}][{c}] out of range [0, 255]"
                )
            grid[r, c] = pixel
    x_hat = grid / 255.0

    # Uniform sigma_hat; inflated by noise + extra slack for unmodeled error.
    sigma_hat = np.full((h, w), max(instance.noise_sigma * 3.0, 0.05), dtype=np.float64)
    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


class MRIKneeLLMAdapter(EnvAdapter):
    env_name: str = "mri-knee-reconstruction"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)


__all__ = ["MRIKneeLLMAdapter", "SYSTEM_PROMPT"]
