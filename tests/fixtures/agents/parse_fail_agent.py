"""Test fixture: returns a well-shaped dict with garbage keys.

solve() succeeds; the env's adapter.parse_response then rejects the
output, producing FailureType.PARSE_ERROR. Exercises the
"deterministic failure → no retry" branch of the M4 retry policy.
"""
from __future__ import annotations

from typing import Any

AGENT_NAME = "parse_fail"


def solve(observation: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    return {"garbage": "not-valid", "answer_text": "{}"}
