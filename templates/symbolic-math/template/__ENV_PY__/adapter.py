"""LLM adapter for __ENV_ID__.

The adapter shapes how the env exposes itself to text-driven solvers:

- ``SYSTEM_PROMPT`` — sets the LLM's role and response format.
- ``build_user_prompt(instance)`` — turns the env's :class:`Instance`
  into the text the LLM reads (the natural-language problem).
- ``parse_response(text, instance)`` — turns the LLM's text reply into
  a :class:`Prediction` that the scorer can evaluate.

The expected JSON envelope is::

    {"answer": "<sympy-parseable string>", "confidence": <float in [0, 1]>}

Parsing is permissive: malformed responses produce a ``Prediction``
with empty ``answer_expr`` and zero confidence, which scores zero on
all reward components. This keeps the env runnable end-to-end even on
the first NotImplementedError-free pass.
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import Instance, Prediction


SYSTEM_PROMPT = (
    "You are an expert solver for the __ENV_ID__ symbolic-math problem "
    "(__DOMAIN__). Given the problem statement, return a JSON object "
    "with your final answer as a SymPy-parseable string and a "
    "self-reported confidence in [0, 1]."
    "\n\nNo prose, no markdown fences — output only the JSON."
)

# Tolerant fence regex used by `parse_response` to recover JSON from
# replies that wrap their content in ```json ... ``` despite the
# system-prompt instruction.
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def build_user_prompt(instance: Instance) -> str:
    """Render the env instance into LLM-readable text."""
    return (
        "PROBLEM:\n"
        + str(instance.prompt)
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<sympy-parseable string>",\n'
        + ' "confidence": <float in [0, 1]>}'
        + "\n\nRespond with the JSON object only."
    )


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract the first plausible JSON object from ``text``.

    Tolerates markdown fences, leading prose, and trailing whitespace.
    Returns ``None`` if no JSON object can be recovered.
    """
    cleaned = text.strip()
    # Try a fenced block first.
    m = _FENCED.search(cleaned)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    # Try the whole string as JSON.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    # Last-resort: find the first balanced { ... } and try that.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_response(text: str, instance: Instance) -> Prediction:
    """Parse the LLM's text into a :class:`Prediction`.

    Permissive: malformed inputs yield an empty-answer prediction
    (zero reward) rather than raising, so the scaffold doesn't crash
    on first contact with a noisy LLM.
    """
    del instance  # not needed for symbolic-math (no shape to match)
    data = _extract_json_block(text)
    if data is None:
        return Prediction(answer_expr="", raw=text, confidence=0.0)

    answer_expr = str(data.get("answer", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return Prediction(
        answer_expr=answer_expr,
        raw=text,
        confidence=confidence,
    )


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
