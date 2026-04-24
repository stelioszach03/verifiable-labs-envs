"""LLM adapter for the phase-retrieval environment."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.phase_retrieval import Instance, Prediction
from verifiable_labs_envs.solvers.adapters._common import (
    coerce_float,
    coerce_int,
    extract_json_block,
    require_key,
    require_list_of_length,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolverError

SYSTEM_PROMPT = """You are an expert at phase retrieval (compressed-sensing from magnitude-only measurements).

Given the *magnitude* of subsampled Fourier coefficients of an unknown real-valued
signal, recover the signal. The phase of each Fourier coefficient is lost — only
|F(x)| is observed. Because |F(-x)| = |F(x)|, recovery is unique only up to a
global sign flip (the scorer checks both x_hat and -x_hat and keeps the better).

You must:
1. Identify exactly k indices where the signal is non-zero (the "support").
2. Estimate the real-valued amplitude at each support entry.

Inputs (all integer-scaled by 1000 for compactness):
- n: total signal length.
- k: exact number of non-zero entries.
- sigma_x1000: magnitude-domain noise std × 1000.
- mask: the m observed frequency indices (sorted ascending).
- y_mag_x1000: m real non-negative magnitudes |F(x)_mask_i| × 1000.

Output exactly one JSON object matching the schema in the user message. No
prose, no markdown fences, no explanations. Example (schematic, k=3):
{"support_idx": [5, 17, 28], "support_amp_x1000": [750, -320, 1100]}

Tip: Gerchberg-Saxton iteration (alternate between phase guess and k-sparse
projection) is the classical baseline — but you can reason your way to a
solution directly from the magnitudes if you see structure."""


def _encode_inputs(instance: Instance) -> str:
    payload = {
        "n": int(instance.n),
        "k": int(instance.k),
        "sigma_x1000": int(round(instance.sigma * 1000)),
        "mask": [int(i) for i in instance.mask.tolist()],
        "y_mag_x1000": [int(round(float(v) * 1000)) for v in instance.y.tolist()],
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

    order = np.argsort(support_indices)
    support_arr = np.asarray([support_indices[i] for i in order], dtype=np.int64)
    amp_arr = np.asarray([amplitudes[i] for i in order], dtype=np.float64)

    x_hat = np.zeros(instance.n, dtype=np.float64)
    x_hat[support_arr] = amp_arr

    # Off-support prior scale = 1.0 (unit signal amplitude prior);
    # on-support: 4σ is a rough ballpark (the propagation from magnitude-domain
    # sigma to signal-domain sigma is not closed-form like ista_baseline's LS).
    sigma_hat = np.full(instance.n, 1.0, dtype=np.float64)
    sigma_hat[support_arr] = max(instance.sigma * 4.0, 0.1)

    return Prediction(x_hat=x_hat, sigma_hat=sigma_hat, support_hat=support_arr)


class PhaseRetrievalLLMAdapter(EnvAdapter):
    env_name: str = "phase-retrieval"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)


__all__ = ["PhaseRetrievalLLMAdapter", "SYSTEM_PROMPT"]
