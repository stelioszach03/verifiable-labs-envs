"""LLM adapter for __ENV_ID__.

Permissive prompt rendering + parsing. The expected JSON envelope on
the final non-tool turn is::

    {"answer": <result>, "confidence": <float in [0, 1]>}

The scorer reads the workspace state directly; the JSON envelope is
advisory + supplies the confidence signal.
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import ToolCallingInstance, ToolCallingPrediction
from __ENV_PY__.tools import init_state

SYSTEM_PROMPT = (
    "You are an agent that completes tasks by composing function "
    "calls. The available tools are described in the JSON-Schema "
    "block accompanying this conversation; emit each tool call via "
    "the standard OpenAI function-call format.\n\n"
    "When the task is complete, emit a final non-tool message of the "
    "form\n"
    '    {"answer": <result>, "confidence": <float in [0, 1]>}\n\n'
    "The scorer reads the workspace state (files, outbox, calculator "
    "history) — the JSON envelope is advisory."
)


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: ToolCallingInstance) -> str:
    seeded_files = ""
    if instance.initial_files:
        listing = "\n".join(
            f"  - {p} ({len(c)} bytes)"
            for p, c in sorted(instance.initial_files.items())
        )
        seeded_files = f"\n\nWORKSPACE FILES:\n{listing}"
    return (
        "PROBLEM:\n"
        f"{instance.prompt}{seeded_files}\n\n"
        f"AVAILABLE TOOLS: {list(instance.available_tools)}\n\n"
        "OUTPUT SCHEMA on the final non-tool turn:\n"
        '{"answer": <result>, "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: ToolCallingInstance) -> ToolCallingPrediction:
    """Parse the LLM's terminating non-tool message.

    Used by ``/v1/score`` (no rollout). For full trajectories, run
    the env's ``run_rollout`` instead.
    """
    confidence = 0.0
    cleaned = text.strip()
    candidates: list[str] = list(_FENCED_RE.findall(cleaned))
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))
    for c in candidates:
        try:
            data: Any = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        confidence = max(0.0, min(1.0, conf))
        break
    return ToolCallingPrediction(
        tool_calls=(),
        final_text=text,
        final_state=init_state(seed=instance.seed, initial_files=instance.initial_files),
        raw=text,
        confidence=confidence,
    )


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
