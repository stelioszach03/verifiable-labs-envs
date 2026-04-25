"""Classical-baseline reference agent.

Each Verifiable Labs env exposes ``env.run_baseline(seed)`` — a
classical solver (OMP for sparse-Fourier, FBP+TV for CT, zero-filled
IFFT for MRI, HIO for phase retrieval, bicubic for super-resolution).
This agent produces a sentinel prediction that the CLI's ``run`` loop
recognises and dispatches to ``env.run_baseline(seed)``, recording
the classical baseline reward as the agent's reward.

Useful as a **reference floor** for ``verifiable compare``: any
learned agent worth deploying should beat this. The baseline reward
is also exposed in every JSONL trace as ``classical_baseline_reward``,
so a single ``simple-baseline`` run isn't strictly necessary — but it
makes the comparison explicit and decouples the baseline from the
learned-agent run.

Usage:

    verifiable run --env sparse-fourier-recovery \\
        --agent examples/agents/simple_baseline_agent.py --n 5 \\
        --out runs/baseline.jsonl
"""
from __future__ import annotations

from typing import Any

AGENT_NAME = "simple-baseline"

# Recognised by ``verifiable run`` — when present, the CLI bypasses the
# adapter parsing path and scores ``env.run_baseline(seed)`` directly.
_SENTINEL_KEY = "__classical_baseline__"


def solve(observation: dict[str, Any]) -> dict[str, Any]:
    return {_SENTINEL_KEY: True}
