"""OpenAI-compatible HTTP agent — example wrapper.

Reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the environment
and calls a chat-completions endpoint. Falls back to a deterministic
fake response when the key is unset — that lets CI exercise the wiring
without spending money.

Configure the model via ``VL_AGENT_MODEL`` (default
``openai/gpt-4o-mini``); configure the temperature via
``VL_AGENT_TEMPERATURE`` (default ``0``).

Usage:

    OPENAI_API_KEY=sk-... \\
    VL_AGENT_MODEL=anthropic/claude-haiku-4.5 \\
    OPENAI_BASE_URL=https://openrouter.ai/api/v1 \\
    verifiable run --env sparse-fourier-recovery \\
        --agent examples/agents/openai_compatible_agent.py --n 3 \\
        --out runs/llm.jsonl
"""
from __future__ import annotations

import os
from typing import Any

from verifiable_labs_envs.agents import OpenAICompatibleAgent

AGENT_NAME = "openai-compatible"


_AGENT: OpenAICompatibleAgent | None = None


def _agent() -> OpenAICompatibleAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = OpenAICompatibleAgent.from_env(
            model=os.environ.get("VL_AGENT_MODEL", "openai/gpt-4o-mini"),
        )
        _AGENT.temperature = float(os.environ.get("VL_AGENT_TEMPERATURE", "0"))
    return _AGENT


def solve(observation: dict[str, Any]) -> dict[str, Any]:
    return _agent().solve(observation)
