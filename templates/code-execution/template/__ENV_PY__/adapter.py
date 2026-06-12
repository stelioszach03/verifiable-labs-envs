"""LLM adapter for __ENV_ID__.

The adapter shapes how the env exposes itself to text-driven solvers:

- ``SYSTEM_PROMPT`` — sets the LLM's role and response format.
- ``build_user_prompt(instance)`` — turns the env's
  :class:`CodeInstance` into the text the LLM reads (signature +
  docstring + visible test block).
- ``parse_response(text, instance)`` — turns the LLM's text reply
  into a :class:`CodePrediction` that the scorer can evaluate.

The expected JSON envelope is::

    {"code": "<Python source string>", "confidence": <float in [0, 1]>}

Parsing is permissive: malformed responses produce a prediction with
empty ``code`` and zero confidence, which scores zero on every
reward component. Keeps the env runnable end-to-end on the first
NotImplementedError-free pass.
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import CodeInstance, CodePrediction

SYSTEM_PROMPT = (
    "You are an expert __DOMAIN__ programmer. Given a function signature "
    "+ docstring + visible test cases, return a JSON object of the form\n\n"
    '    {"code": "<Python source>", "confidence": <float in [0, 1]>}\n\n'
    "where ``code`` is a complete Python source string defining the function "
    "the prompt asks for. No prose, no markdown fences — JSON only."
)

_FENCED_RE = re.compile(r"```(?:python|json)?\s*(.+?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: CodeInstance) -> str:
    """Render the env instance into LLM-readable text."""
    return (
        "PROBLEM:\n"
        f"{instance.prompt}\n\n"
        "OUTPUT SCHEMA:\n"
        '{"code": "<Python source string>", "confidence": <float in [0, 1]>}\n\n'
        "Respond with the JSON object only."
    )


def parse_response(text: str, instance: CodeInstance) -> CodePrediction:
    """Parse the LLM's text into a :class:`CodePrediction`.

    Permissive: malformed inputs yield an empty-code prediction (zero
    reward) rather than raising, so the scaffold doesn't crash on
    first contact with a noisy LLM.
    """
    del instance
    cleaned = text.strip()
    candidates: list[str] = []
    fenced = _FENCED_RE.findall(cleaned)
    candidates.extend(fenced)
    candidates.append(cleaned)
    bare = _JSON_OBJECT_RE.search(cleaned)
    if bare:
        candidates.append(bare.group(0))

    for cand in candidates:
        try:
            data: Any = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        code = str(data.get("code", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return CodePrediction(code=code, raw=text, confidence=confidence)

    return CodePrediction(code="", raw=text, confidence=0.0)


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
