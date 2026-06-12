"""Deterministic stub student for the 29.D eval harness + 29.E API.

29.D ships the eval surface backed by a *stub* predictor that returns
``0.5 + uniform(-0.1, 0.1)`` per :doc:`PHASE_29_PLAN.md` §10. This
lets the harness shape land + integration tests exercise without the
trained student weights (those arrive in 29.G).

The stub is **deterministic** for reproducibility — same prompt +
completion always produces the same scalar — so eval reports are
comparable across runs.

Audit posture: every stub response carries ``schema_version="v0.1.0-stub"``
so downstream consumers (logging, SDK) can detect the placeholder
shape and warn the customer that the response is canned.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SCHEMA_VERSION: str = "v0.1.0-stub"
DEFAULT_REWARD_FLOOR: float = 0.4
DEFAULT_REWARD_CEILING: float = 0.6
DEFAULT_DELTA: float = 0.1
DEFAULT_LATENCY_MS: int = 50
DEFAULT_COVERAGE_GUARANTEE: float = 0.90
DEFAULT_MODEL_ID: str = "vlabs-reward-distilled-qwen-1-5b-v0.1.0"


@dataclass(frozen=True)
class StubScoreResult:
    """Canonical stub-response shape mirroring the 29.E API surface."""

    reward: float
    confidence_interval: tuple[float, float]
    coverage_guarantee: float
    model_id: str
    schema_version: str = DEFAULT_SCHEMA_VERSION
    cache_hit: bool = False
    latency_ms: int = DEFAULT_LATENCY_MS
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "confidence_interval": [
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1]),
            ],
            "coverage_guarantee": float(self.coverage_guarantee),
            "model_id": str(self.model_id),
            "schema_version": str(self.schema_version),
            "cache_hit": bool(self.cache_hit),
            "latency_ms": int(self.latency_ms),
            "metadata": dict(self.metadata),
        }


def stub_score(
    prompt: str,
    response: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    delta: float = DEFAULT_DELTA,
    seed: int | None = None,
) -> StubScoreResult:
    """Score a (prompt, response) pair with the deterministic stub.

    Behaviour:

    - ``reward`` is a deterministic offset of 0.5 in the range
      ``[0.5 - delta, 0.5 + delta]`` derived from the SHA-256 hash of
      ``(prompt, response, seed)`` so identical inputs return
      identical outputs across processes.
    - ``confidence_interval`` is ``[reward - delta, reward + delta]``
      clipped to ``[0, 1]`` — a fixed nominal width.
    - ``coverage_guarantee`` is the locked 0.90 from D10-A so the
      service contract reads correctly even in stub mode.
    - ``cache_hit`` is always False; cache integration ships in 29.E.
    - ``schema_version`` ends in ``-stub`` so callers detect the
      placeholder.
    """
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative; got {delta}")
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"|")
    h.update(response.encode("utf-8"))
    if seed is not None:
        h.update(b"|")
        h.update(str(seed).encode("utf-8"))
    digest = int.from_bytes(h.digest()[:8], "big")
    # Map to [0, 1] then to [-delta, +delta] then to [0.5 - delta, 0.5 + delta].
    fraction = (digest % 1_000_001) / 1_000_000
    offset = (fraction * 2 - 1) * delta
    reward = max(0.0, min(1.0, 0.5 + offset))
    low = max(0.0, reward - delta)
    high = min(1.0, reward + delta)
    return StubScoreResult(
        reward=reward,
        confidence_interval=(low, high),
        coverage_guarantee=DEFAULT_COVERAGE_GUARANTEE,
        model_id=model_id,
    )


def stub_predictor(
    *, delta: float = DEFAULT_DELTA, seed: int | None = None
) -> Callable[[str, str], float]:
    """Adapter for the calibration / eval surface — returns a callable
    matching ``(prompt, completion) -> reward``."""

    def predict(prompt: str, completion: str) -> float:
        return stub_score(prompt, completion, delta=delta, seed=seed).reward

    return predict


def is_stub_payload(payload: dict[str, Any]) -> bool:
    """Predicate: does this look like a stub response?

    Audit / monitoring layers use this to flag canned responses so the
    customer doesn't conflate them with real student predictions.
    """
    schema = str(payload.get("schema_version", ""))
    return schema.endswith("-stub")


def stub_score_floor() -> float:
    """The minimum scalar the stub can return at default ``delta``."""
    return DEFAULT_REWARD_FLOOR


def stub_score_ceiling() -> float:
    """The maximum scalar the stub can return at default ``delta``."""
    return DEFAULT_REWARD_CEILING


__all__ = [
    "DEFAULT_COVERAGE_GUARANTEE",
    "DEFAULT_DELTA",
    "DEFAULT_LATENCY_MS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_REWARD_CEILING",
    "DEFAULT_REWARD_FLOOR",
    "DEFAULT_SCHEMA_VERSION",
    "StubScoreResult",
    "is_stub_payload",
    "stub_predictor",
    "stub_score",
    "stub_score_ceiling",
    "stub_score_floor",
]
