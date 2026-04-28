"""LLM adapter for the super-resolution-div2k-x4 environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skimage import transform as sktransform

from verifiable_labs_envs.envs.super_resolution import Instance, Prediction
from verifiable_labs_envs.solvers.adapters._common import (
    coerce_int,
    extract_json_block,
    require_key,
    require_list_of_length,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolverError

SYSTEM_PROMPT = """You are an expert image reconstruction solver.

You will receive a low-resolution grayscale image that was produced by
applying Gaussian blur and 4x decimation to an unknown high-resolution
source, plus a small amount of additive noise. Pixel values are integers
in [0, 255]. You must return a same-size refined version of the image:
denoise it, sharpen edges, suppress measurement artifacts. Your output
will be bicubic-upsampled server-side to reconstruct the high-resolution
image, so your job is to recover the best possible low-resolution prior.

Output exactly one JSON object matching the schema given in the user
message. No prose, no markdown fences. Example of the output format
(schematic for a 2x2 image): {"image": [[128, 255], [0, 64]]}"""


def _encode_inputs(instance: Instance) -> str:
    # Encode measurement as a 2D int grid in [0, 255].
    y = np.clip(instance.y, 0.0, 1.0)
    y_u8 = (y * 255.0).round().astype(np.int64)
    payload = {
        "n_rows": int(y.shape[0]),
        "n_cols": int(y.shape[1]),
        "noise_sigma_x1000": int(round(instance.noise_sigma * 1000)),
        "blur_sigma_x100": int(round(instance.blur_sigma * 100)),
        "factor": int(instance.factor),
        "image": [[int(v) for v in row] for row in y_u8.tolist()],
    }
    return json.dumps(payload, separators=(",", ":"))


def _build_user_prompt(instance: Instance) -> str:
    inputs_json = _encode_inputs(instance)
    lr_rows, lr_cols = instance.y.shape
    return (
        "INPUTS:\n"
        + inputs_json
        + "\n\n"
        + "OUTPUT SCHEMA:\n"
        + "{\"image\": ["
        + f"{lr_rows} lists, each with {lr_cols} integers in [0, 255]"
        + "]}\n\n"
        + "Respond with the JSON object only."
    )


def _parse_response(text: str, instance: Instance) -> Prediction:
    parsed = extract_json_block(text)
    image_raw = require_key(parsed, "image")

    lr_rows, lr_cols = instance.y.shape

    rows = require_list_of_length(image_raw, lr_rows, "image", lenient=True)
    grid = np.empty((lr_rows, lr_cols), dtype=np.float64)
    for r, row_raw in enumerate(rows):
        row = require_list_of_length(row_raw, lr_cols, f"image[{r}]", lenient=True)
        for c, value in enumerate(row):
            pixel = coerce_int(value, f"image[{r}][{c}]")
            if not 0 <= pixel <= 255:
                raise LLMSolverError(
                    f"pixel {pixel} at image[{r}][{c}] out of range [0, 255]"
                )
            grid[r, c] = pixel

    lr_image = grid / 255.0
    x_hat = sktransform.resize(
        lr_image, instance.shape, order=3, anti_aliasing=False, mode="reflect"
    ).astype(np.float64)

    # Edge-weighted sigma_hat — same heuristic used by bicubic_baseline.
    from scipy.ndimage import sobel

    gx = sobel(x_hat, axis=0, mode="reflect")
    gy = sobel(x_hat, axis=1, mode="reflect")
    grad = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad / (float(grad.max()) + 1e-8)
    sigma_hat = 2.0 * float(instance.noise_sigma) + 0.20 * grad_norm
    sigma_hat = np.maximum(sigma_hat, 1e-4)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat)


class SuperResolutionLLMAdapter(EnvAdapter):
    env_name: str = "super-resolution-div2k-x4"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)
