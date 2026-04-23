"""LLM adapter for the lodopab-ct-simplified environment.

The LLM is given a rough filtered-back-projection reconstruction at a
coarse 32x32 resolution (downsampled from the native phantom shape) and
asked to refine it — suppress streak artifacts, sharpen edges, improve
contrast. Its 32x32 output is bicubic-upsampled to the phantom shape
before scoring. This framing lets an LLM contribute without requiring
it to invert the Radon operator from scratch (which would fail on all
models in this budget tier).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skimage import transform as sktransform

from verifiable_labs_envs.envs.lodopab_ct import Instance, Prediction
from verifiable_labs_envs.forward_ops import radon_fbp
from verifiable_labs_envs.solvers.adapters._common import (
    coerce_int,
    extract_json_block,
    require_key,
    require_list_of_length,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolverError

COARSE_SIZE: int = 32  # side length of the input/output grid shown to the LLM

SYSTEM_PROMPT = """You are an expert at CT (computed tomography) image reconstruction.

You will receive a rough filtered-back-projection (FBP) reconstruction
of a 2D phantom at a coarse 32x32 resolution. The full reconstruction
problem is parallel-beam CT with 60 projection angles (under-sampled),
with Gaussian noise on the sinogram. FBP shows characteristic streak
artifacts and edge ringing; your task is to refine the 32x32 image:
suppress streaks, sharpen organ/phantom boundaries, restore contrast.
Your 32x32 output will be bicubic-upsampled server-side to the native
phantom resolution before scoring.

Pixel values are integers in [0, 255]. Output exactly one JSON object
matching the user schema. No prose, no markdown fences. Example
(schematic for 2x2): {"image": [[30, 210], [45, 180]]}"""


def _downsample_to_coarse(img: np.ndarray, side: int) -> np.ndarray:
    """Bicubic resize to `(side, side)` in [0, 1]."""
    return sktransform.resize(
        img, (side, side), order=3, anti_aliasing=True, mode="reflect"
    ).astype(np.float64)


def _encode_inputs(instance: Instance) -> str:
    # Produce the FBP reconstruction, downsample to COARSE_SIZE, encode as uint8.
    fbp = radon_fbp(instance.y, instance.angles_deg, output_size=instance.shape[0])
    fbp = np.clip(fbp, 0.0, 1.0)
    coarse = _downsample_to_coarse(fbp, COARSE_SIZE)
    coarse_u8 = (coarse * 255.0).round().astype(np.int64)
    payload = {
        "shape": [COARSE_SIZE, COARSE_SIZE],
        "noise_sigma_x100": int(round(instance.noise_sigma * 100)),
        "n_angles": int(instance.n_angles),
        "image": [[int(v) for v in row] for row in coarse_u8.tolist()],
    }
    return json.dumps(payload, separators=(",", ":"))


def _build_user_prompt(instance: Instance) -> str:
    inputs_json = _encode_inputs(instance)
    return (
        "INPUTS (rough FBP reconstruction at 32x32):\n"
        + inputs_json
        + "\n\n"
        + "OUTPUT SCHEMA:\n"
        + "{\"image\": ["
        + f"{COARSE_SIZE} lists, each with {COARSE_SIZE} integers in [0, 255]"
        + "]}\n\n"
        + "Respond with the JSON object only."
    )


def _parse_response(text: str, instance: Instance) -> Prediction:
    parsed = extract_json_block(text)
    image_raw = require_key(parsed, "image")

    rows = require_list_of_length(image_raw, COARSE_SIZE, "image")
    grid = np.empty((COARSE_SIZE, COARSE_SIZE), dtype=np.float64)
    for r, row_raw in enumerate(rows):
        row = require_list_of_length(row_raw, COARSE_SIZE, f"image[{r}]")
        for c, value in enumerate(row):
            pixel = coerce_int(value, f"image[{r}][{c}]")
            if not 0 <= pixel <= 255:
                raise LLMSolverError(
                    f"pixel {pixel} at image[{r}][{c}] out of range [0, 255]"
                )
            grid[r, c] = pixel

    coarse_image = grid / 255.0
    x_hat = sktransform.resize(
        coarse_image, instance.shape, order=3, anti_aliasing=False, mode="reflect"
    ).astype(np.float64)

    from scipy.ndimage import sobel

    gx = sobel(x_hat, axis=0, mode="reflect")
    gy = sobel(x_hat, axis=1, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad / (float(grad.max()) + 1e-8)
    sigma_hat = 0.02 + 2.0 * float(instance.noise_sigma) * (0.2 + grad_norm)
    sigma_hat = np.maximum(sigma_hat, 1e-4)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


class LodopabCtLLMAdapter(EnvAdapter):
    env_name: str = "lodopab-ct-simplified"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)
