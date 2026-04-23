"""LLM adapter for the sparse-fourier-recovery environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.sparse_fourier import Instance, Prediction
from verifiable_labs_envs.solvers.adapters._common import (
    coerce_float,
    coerce_int,
    extract_json_block,
    require_key,
    require_list_of_length,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolverError

SYSTEM_PROMPT = """You are an expert at sparse signal recovery (compressed sensing).

Given subsampled Fourier measurements of an unknown real-valued signal, you must:
1. Identify exactly k indices where the signal is non-zero (the "support").
2. Estimate the real-valued amplitude at each support entry.

Inputs are integers scaled by 1000 for compactness:
- n: total signal length.
- k: exact number of non-zero entries.
- sigma_x1000: noise standard deviation times 1000.
- mask: the m observed frequency indices (sorted ascending).
- y_re_x1000, y_im_x1000: real and imaginary parts of the m complex
  Fourier coefficients at those frequencies, each times 1000.

Output exactly one JSON object matching the schema given in the user
message. Output nothing else - no prose, no markdown fences, no
explanations. Example of the output format (schematic, k=3):
{"support_idx": [5, 17, 42], "support_amp_x1000": [750, -320, 1100]}"""


def _encode_inputs(instance: Instance) -> str:
    sigma_x1000 = int(round(instance.sigma * 1000))
    mask_list = [int(i) for i in instance.mask]
    y_re = [int(round(v * 1000)) for v in instance.y.real.tolist()]
    y_im = [int(round(v * 1000)) for v in instance.y.imag.tolist()]
    payload = {
        "n": int(instance.n),
        "k": int(instance.k),
        "sigma_x1000": sigma_x1000,
        "mask": mask_list,
        "y_re_x1000": y_re,
        "y_im_x1000": y_im,
    }
    return json.dumps(payload, separators=(", ", ": "))


def _build_user_prompt(instance: Instance) -> str:
    inputs_json = _encode_inputs(instance)
    return (
        "INPUTS:\n"
        + inputs_json
        + "\n\n"
        + "OUTPUT SCHEMA:\n"
        + '{"support_idx": ['
        + f"{instance.k} integers in [0, {instance.n}), sorted ascending],\n"
        + ' "support_amp_x1000": ['
        + f"{instance.k} integers, real amplitudes x 1000, same order as support_idx]"
        + "}\n\n"
        + "Respond with the JSON object only."
    )


def _ls_sigma_hat_on_support(
    mask: np.ndarray, support: np.ndarray, n: int, sigma: float
) -> np.ndarray:
    """LS-theoretic per-entry std on the given support, matching `ista_baseline`."""
    if support.size == 0:
        return np.array([], dtype=np.float64)
    a_s = np.exp(-2j * np.pi * np.outer(mask, support) / n) / np.sqrt(n)
    a_stacked = np.vstack([a_s.real, a_s.imag])
    try:
        cov = (float(sigma) ** 2 / 2.0) * np.linalg.inv(a_stacked.T @ a_stacked)
        sigma_s = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        sigma_s = np.full(support.size, float(sigma), dtype=np.float64)
    return np.maximum(sigma_s, 1e-6)


def _parse_response(text: str, instance: Instance) -> Prediction:
    parsed = extract_json_block(text)

    support_raw = require_key(parsed, "support_idx")
    amp_raw = require_key(parsed, "support_amp_x1000")

    support_list = require_list_of_length(support_raw, instance.k, "support_idx")
    amp_list = require_list_of_length(amp_raw, instance.k, "support_amp_x1000")

    support_indices: list[int] = []
    for pos, value in enumerate(support_list):
        idx = coerce_int(value, f"support_idx[{pos}]")
        if not 0 <= idx < instance.n:
            raise LLMSolverError(
                f"support index {idx} at position {pos} out of range [0, {instance.n})"
            )
        support_indices.append(idx)

    if len(set(support_indices)) != len(support_indices):
        dupes = sorted({i for i in support_indices if support_indices.count(i) > 1})
        raise LLMSolverError(f"duplicate support indices: {dupes}")

    amplitudes: list[float] = []
    for pos, value in enumerate(amp_list):
        amp = coerce_float(value, f"support_amp_x1000[{pos}]")
        amplitudes.append(amp / 1000.0)

    # Align into the instance's index space and build Prediction.
    order = np.argsort(support_indices)
    support_arr = np.asarray([support_indices[i] for i in order], dtype=np.int64)
    amp_arr = np.asarray([amplitudes[i] for i in order], dtype=np.float64)

    x_hat = np.zeros(instance.n, dtype=np.float64)
    x_hat[support_arr] = amp_arr

    # Signal-amplitude prior scale off-support (matches ista_baseline behaviour).
    sigma_hat = np.full(instance.n, 1.0, dtype=np.float64)
    sigma_hat[support_arr] = _ls_sigma_hat_on_support(
        instance.mask, support_arr, instance.n, instance.sigma
    )

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat, support_hat=support_arr)


class SparseFourierLLMAdapter(EnvAdapter):
    env_name: str = "sparse-fourier-recovery"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)
