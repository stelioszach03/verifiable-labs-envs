"""Test fixture: raises on the first call per seed, succeeds on subsequent calls.

Used to verify that ``--max-retries N`` retries non-deterministic
failures and reports ``retries=K`` on the final (successful) trace.

State persists in an on-disk file so tests can isolate runs by
pointing ``FLAKY_AGENT_STATE`` at a tmp_path. If the env var is unset
the file falls back to ``/tmp/flaky_agent_state.txt`` — convenient for
hand-running examples but NOT used by the unit tests (which always
override).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

AGENT_NAME = "flaky"


def _state_path() -> Path:
    return Path(os.environ.get("FLAKY_AGENT_STATE", "/tmp/flaky_agent_state.txt"))


def _seen_seeds() -> set[str]:
    p = _state_path()
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def _record_seed(seed_str: str) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(seed_str + "\n")


def solve(observation: dict[str, Any]) -> dict[str, Any]:
    seed = observation.get("seed", 0)
    seed_str = str(seed)
    if seed_str not in _seen_seeds():
        _record_seed(seed_str)
        raise RuntimeError(f"flaky first call for seed={seed_str}")

    # On retry: emit a valid zero-prediction so the episode scores cleanly.
    inputs = observation.get("inputs") or {}
    k = int(inputs.get("k", 10))
    n = int(inputs.get("n", k))
    return {
        "support_idx": sorted(range(min(k, n))),
        "support_amp_x1000": [0] * k,
    }
