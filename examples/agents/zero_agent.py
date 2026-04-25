"""Zero-amplitude agent — the simplest possible agent.

Returns a prediction with all amplitudes zero, so every env that
expects a sparse-recovery shape will mark it ``parse_ok=True`` (the
JSON is well-formed) but score it near zero. Useful as a CI smoke
target and as a floor for ``verifiable compare``.

Usage:

    verifiable run --env sparse-fourier-recovery \\
        --agent examples/agents/zero_agent.py --n 3 --out runs/zero.jsonl
"""
from __future__ import annotations

from typing import Any

AGENT_NAME = "zero"


def solve(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a zero-amplitude prediction matching the env's schema.

    For sparse-recovery envs (sparse-fourier, phase-retrieval) the
    schema is ``{support_idx, support_amp_x1000}`` of length ``k``; for
    image envs (CT, MRI, super-resolution) the schema is
    ``{image_x255, uncertainty_x255}`` of length ``H * W``.

    We dispatch on whichever schema the observation hints at — the
    env-specific instance fields are forwarded under
    ``observation["inputs"]`` by the CLI's ``run`` subcommand.
    """
    inputs = observation.get("inputs") or {}

    # Sparse-recovery shape: emit zeros for k support points.
    if "k" in inputs:
        k = int(inputs["k"])
        n = int(inputs.get("n", k))
        # Pick the first k indices in [0, n) — guarantees parse_ok=True.
        support = sorted(range(min(k, n)))
        if len(support) < k:
            support.extend([n - 1] * (k - len(support)))
        return {
            "support_idx": support,
            "support_amp_x1000": [0] * k,
        }

    # Image shape: emit a uniformly grey image with low uncertainty.
    if "h" in inputs and "w" in inputs:
        h, w = int(inputs["h"]), int(inputs["w"])
        n_pixels = h * w
        return {
            "image_x255": [128] * n_pixels,
            "uncertainty_x255": [10] * n_pixels,
        }

    # Fallback: emit a generic empty prediction. The env's parser will
    # reject it; this still records a parse-fail trace.
    return {"answer_text": "{}"}
