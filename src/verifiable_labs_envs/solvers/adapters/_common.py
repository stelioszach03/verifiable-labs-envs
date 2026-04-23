"""Shared helpers for LLM-adapter JSON parsing."""
from __future__ import annotations

import json
import re
from typing import Any

from verifiable_labs_envs.solvers.llm_solver import LLMSolverError

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(text: str) -> dict[str, Any]:
    """Pull a single JSON object out of an LLM response.

    Handles the three common wrappers LLMs inject around JSON:
    1. Plain JSON (the ideal case).
    2. Triple-backtick fenced code blocks (``` or ```json ... ```).
    3. Prose before/after a JSON object.

    Uses a balanced-brace scan to extract the first top-level ``{...}`` block,
    so prose like "Here is my answer: {...}." is tolerated.

    Raises ``LLMSolverError`` if no JSON object is found or parsing fails.
    """
    if not text:
        raise LLMSolverError("empty response text")

    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text

    start = candidate.find("{")
    if start == -1:
        raise LLMSolverError("no JSON block found in response (no '{' character)")

    depth = 0
    end = -1
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise LLMSolverError("no JSON block found in response (unbalanced braces)")

    blob = candidate[start : end + 1]
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise LLMSolverError(f"JSON parse error: {exc.msg} at position {exc.pos}") from exc

    if not isinstance(parsed, dict):
        raise LLMSolverError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def require_key(obj: dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise LLMSolverError(f"missing key '{key}' in response")
    return obj[key]


def require_list_of_length(obj: Any, expected_len: int, name: str) -> list[Any]:
    if not isinstance(obj, list):
        raise LLMSolverError(f"'{name}' must be a list, got {type(obj).__name__}")
    if len(obj) != expected_len:
        raise LLMSolverError(f"expected {expected_len} entries in '{name}', got {len(obj)}")
    return obj


def coerce_int(value: Any, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMSolverError(f"non-integer value in '{name}': {value!r}") from exc
    return out


def coerce_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMSolverError(f"non-numeric value in '{name}': {value!r}") from exc
    import math

    if math.isnan(out) or math.isinf(out):
        raise LLMSolverError(f"non-finite value in '{name}': {value!r}")
    return out
