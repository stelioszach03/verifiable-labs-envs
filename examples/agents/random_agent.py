"""Random-prediction agent — a stronger zero-info baseline than ``zero``.

Samples random support indices and amplitudes from sane priors so the
distribution of rewards has non-zero variance. Use as a sanity check
that ``verifiable compare`` actually distinguishes agents.

Deterministic per call: the seed is derived from the observation hash
so reruns reproduce.

Usage:

    verifiable run --env sparse-fourier-recovery \\
        --agent examples/agents/random_agent.py --n 5 --out runs/random.jsonl
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

AGENT_NAME = "random"


def _seed_from_observation(observation: dict[str, Any]) -> int:
    blob = json.dumps(observation, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def solve(observation: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(_seed_from_observation(observation))
    inputs = observation.get("inputs") or {}

    if "k" in inputs:
        k = int(inputs["k"])
        n = int(inputs.get("n", k))
        # Random support of size k drawn without replacement from [0, n).
        support = sorted(rng.sample(range(n), k))
        # Random amplitudes in a typical sparse-Fourier scale (±1.5).
        amps = [int(round(rng.uniform(-1.5, 1.5) * 1000)) for _ in range(k)]
        return {"support_idx": support, "support_amp_x1000": amps}

    if "h" in inputs and "w" in inputs:
        h, w = int(inputs["h"]), int(inputs["w"])
        n_pixels = h * w
        return {
            "image_x255": [rng.randint(0, 255) for _ in range(n_pixels)],
            "uncertainty_x255": [rng.randint(5, 30) for _ in range(n_pixels)],
        }

    return {"answer_text": "{}"}
