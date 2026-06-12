"""Stub PRM inference for the 30.D eval harness + 30.E API.

30.D ships the eval surface backed by a *stub* PRM that returns a
deterministic per-step ``0.5 ± 0.1`` sequence per
:doc:`PHASE_30_PLAN.md` §10. This lets the harness shape land +
integration tests exercise without trained student weights (those
arrive in 30.G).

The stub is **deterministic** for reproducibility — same
``(prompt, step_index, step_text)`` tuple always produces the same
scalar — so eval reports are comparable across runs and the SHA-256
audit hashes line up.

Audit posture: every stub response carries
``schema_version="v0.1.0-stub"`` so downstream consumers (logging,
SDK, monitoring) can detect the placeholder shape and warn the
customer that the real model isn't online yet.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SCHEMA_VERSION: str = "v0.1.0-stub"
DEFAULT_DELTA: float = 0.1
"""Stub CI half-width — matches Phase 29 distilled RM stub for shape
consistency."""

DEFAULT_REWARD_FLOOR: float = 0.4
DEFAULT_REWARD_CEILING: float = 0.6
DEFAULT_LATENCY_MS: int = 80
"""Per :doc:`PHASE_30_PLAN.md` §10 — stub latency (slightly higher
than Phase 29's 50 ms because per-step output is denser)."""

DEFAULT_COVERAGE_GUARANTEE: float = 0.90
DEFAULT_MODEL_ID: str = "vlabs-prm-distilled-qwen-1-5b-v0.1.0"


@dataclass(frozen=True)
class StubProcessScoreResult:
    """Canonical stub-response shape mirroring the 30.E API surface."""

    step_rewards: tuple[float, ...]
    step_confidence_intervals: tuple[tuple[float, float], ...]
    aggregate_reward: float
    aggregate_confidence_interval: tuple[float, float]
    coverage_guarantee: float
    step_count: int
    model_id: str
    schema_version: str = DEFAULT_SCHEMA_VERSION
    cache_hit: bool = False
    latency_ms: int = DEFAULT_LATENCY_MS
    segmentation_warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_rewards": [float(r) for r in self.step_rewards],
            "step_confidence_intervals": [
                [float(ci[0]), float(ci[1])]
                for ci in self.step_confidence_intervals
            ],
            "aggregate_reward": float(self.aggregate_reward),
            "aggregate_confidence_interval": [
                float(self.aggregate_confidence_interval[0]),
                float(self.aggregate_confidence_interval[1]),
            ],
            "coverage_guarantee": float(self.coverage_guarantee),
            "step_count": int(self.step_count),
            "model_id": str(self.model_id),
            "schema_version": str(self.schema_version),
            "cache_hit": bool(self.cache_hit),
            "latency_ms": int(self.latency_ms),
            "segmentation_warning": self.segmentation_warning,
            "metadata": dict(self.metadata),
        }


def stub_process_score(
    prompt: str,
    steps: Sequence[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    delta: float = DEFAULT_DELTA,
    seed: int | None = None,
    segmentation_warning: str | None = None,
) -> StubProcessScoreResult:
    """Score a (prompt, segmented-trace) pair with the deterministic stub.

    Behaviour:

    - Each step's reward is a deterministic offset of 0.5 in
      ``[0.5 - delta, 0.5 + delta]`` derived from the SHA-256 hash of
      ``(prompt, step_index, step_text, seed)``.
    - Each step's confidence interval is
      ``[reward - delta, reward + delta]`` clipped to ``[0, 1]``.
    - Aggregate reward = arithmetic mean of step rewards.
    - Aggregate CI = mean ± delta clipped to ``[0, 1]``.
    - ``coverage_guarantee`` is the locked 0.90 from D9.
    - ``schema_version`` ends in ``-stub`` so callers detect the
      placeholder.
    """
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative; got {delta}")
    if not steps:
        raise ValueError("steps must be non-empty")

    rewards: list[float] = []
    cis: list[tuple[float, float]] = []
    for i, step_text in enumerate(steps):
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"|")
        h.update(str(i).encode("utf-8"))
        h.update(b"|")
        h.update(step_text.encode("utf-8"))
        if seed is not None:
            h.update(b"|")
            h.update(str(seed).encode("utf-8"))
        digest = int.from_bytes(h.digest()[:8], "big")
        fraction = (digest % 1_000_001) / 1_000_000
        offset = (fraction * 2 - 1) * delta
        reward = _clip01(0.5 + offset)
        low = max(0.0, reward - delta)
        high = min(1.0, reward + delta)
        rewards.append(reward)
        cis.append((low, high))

    aggregate = sum(rewards) / len(rewards)
    aggregate_ci = (
        max(0.0, aggregate - delta),
        min(1.0, aggregate + delta),
    )
    return StubProcessScoreResult(
        step_rewards=tuple(rewards),
        step_confidence_intervals=tuple(cis),
        aggregate_reward=aggregate,
        aggregate_confidence_interval=aggregate_ci,
        coverage_guarantee=DEFAULT_COVERAGE_GUARANTEE,
        step_count=len(steps),
        model_id=model_id,
        segmentation_warning=segmentation_warning,
    )


# ── adapters for the eval / calibration surfaces ───────────────────


def stub_step_predictor(
    *, delta: float = DEFAULT_DELTA, seed: int | None = None
) -> Callable[[str, Sequence[str], int], float]:
    """Return a per-step callable matching
    ``(prompt, steps, step_index) -> reward``.

    Used by 30.D's calibration + eval surfaces — the trained student
    in 30.G will replace this with a real per-step inference call.
    """

    def predict(prompt: str, steps: Sequence[str], step_index: int) -> float:
        if not (0 <= step_index < len(steps)):
            raise IndexError(
                f"step_index {step_index} out of range for {len(steps)} steps"
            )
        result = stub_process_score(
            prompt, steps, delta=delta, seed=seed
        )
        return result.step_rewards[step_index]

    return predict


def stub_aggregate_predictor(
    *, delta: float = DEFAULT_DELTA, seed: int | None = None
) -> Callable[[str, Sequence[str]], float]:
    """Return a trace-level callable matching
    ``(prompt, steps) -> aggregate_reward``."""

    def predict(prompt: str, steps: Sequence[str]) -> float:
        return stub_process_score(prompt, steps, delta=delta, seed=seed).aggregate_reward

    return predict


def stub_full_predictor(
    *, delta: float = DEFAULT_DELTA, seed: int | None = None
) -> Callable[[str, Sequence[str]], StubProcessScoreResult]:
    """Return the full-shape callable matching
    ``(prompt, steps) -> StubProcessScoreResult``."""

    def predict(prompt: str, steps: Sequence[str]) -> StubProcessScoreResult:
        return stub_process_score(prompt, steps, delta=delta, seed=seed)

    return predict


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ── audit helpers ──────────────────────────────────────────────────


def is_stub_payload(payload: dict[str, Any]) -> bool:
    """Predicate: does this look like a stub PRM response?

    Audit / monitoring layers use this to flag canned responses so
    customers don't conflate them with real student predictions.
    """
    schema = str(payload.get("schema_version", ""))
    return schema.endswith("-stub")


def stub_score_floor() -> float:
    return DEFAULT_REWARD_FLOOR


def stub_score_ceiling() -> float:
    return DEFAULT_REWARD_CEILING


__all__ = [
    "DEFAULT_COVERAGE_GUARANTEE",
    "DEFAULT_DELTA",
    "DEFAULT_LATENCY_MS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_REWARD_CEILING",
    "DEFAULT_REWARD_FLOOR",
    "DEFAULT_SCHEMA_VERSION",
    "StubProcessScoreResult",
    "is_stub_payload",
    "stub_aggregate_predictor",
    "stub_full_predictor",
    "stub_process_score",
    "stub_score_ceiling",
    "stub_score_floor",
    "stub_step_predictor",
]
