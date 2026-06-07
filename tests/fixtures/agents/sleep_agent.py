"""Test fixture: sleeps 120 seconds in solve(). Used to exercise --timeout-seconds.

The 120-second budget is far longer than any reasonable test timeout, so
SIGALRM will always interrupt before the sleep completes naturally.
"""
from __future__ import annotations

import time
from typing import Any

AGENT_NAME = "sleep"


def solve(observation: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    time.sleep(120)
    return {"never": "reached"}
