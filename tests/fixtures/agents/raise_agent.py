"""Test fixture: raises ValueError in solve(). Exercises generic-exception path."""
from __future__ import annotations

from typing import Any

AGENT_NAME = "raise"


def solve(observation: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    raise ValueError("test exception from raise_agent")
