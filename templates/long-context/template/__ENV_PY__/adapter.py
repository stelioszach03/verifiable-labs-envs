"""LLM adapter for __ENV_ID__.

Permissive prompt rendering + parsing. The expected JSON envelope is::

    {"answer": "<extracted text>", "confidence": <float in [0, 1]>}

The scorer reads ``answer`` field; the JSON envelope is mandatory
(missing or malformed → ``format_valid = 0``).
"""
from __future__ import annotations

import json
import re
from typing import Any

from __ENV_PY__.data import NeedleInstance, NeedlePrediction

SYSTEM_PROMPT = (
    "You are a careful long-context reader. The user message contains "
    "a multi-document corpus separated by ``---DOCUMENT N: <title>---`` "
    "headers, followed by a question. Locate the relevant fact and "
    "return it.\n\n"
    "Output exactly one JSON object of the form\n"
    '    {"answer": "<extracted text>", "confidence": <float in [0, 1]>}\n\n'
    "No prose, no markdown fences — JSON only."
)


_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_prompt(instance: NeedleInstance) -> str:
    body = instance.corpus.render_prompt(question=instance.question)
    return (
        body
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<extracted text>", "confidence": <float in [0, 1]>}'
    )


def parse_response(text: str, instance: NeedleInstance) -> NeedlePrediction:
    """Parse the LLM's text into a :class:`NeedlePrediction`."""
    del instance
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
        answer = str(data.get("answer", "")).strip()
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return NeedlePrediction(answer=answer, raw=text, confidence=confidence)
    return NeedlePrediction(answer="", raw=text, confidence=0.0)


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "parse_response"]
