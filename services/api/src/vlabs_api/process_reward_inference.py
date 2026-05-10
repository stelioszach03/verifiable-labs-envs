"""Service layer for the distilled PRM endpoints (Phase 30.E).

Wraps :func:`verifiable_labs_envs.process_reward.inference.stub_process_score`
in the API surface — adds prompt + trace hashing (D11 / R11 privacy
posture), Redis cache key derivation, segmentation, and audit-row
construction. The trained-student inference layer arrives in 30.G;
until then every score returns the canonical stub payload with
``schema_version="v0.1.0-stub"``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verifiable_labs_envs.process_reward.dataset import DEFAULT_MAX_STEPS
from verifiable_labs_envs.process_reward.inference import (
    DEFAULT_COVERAGE_GUARANTEE,
    DEFAULT_SCHEMA_VERSION,
    StubProcessScoreResult,
    stub_process_score,
)
from verifiable_labs_envs.process_reward.segmentation import (
    SegmentationOutcome,
    segment_trace,
)

from vlabs_api.db import ProcessRewardModel
from vlabs_api.errors import (
    ProcessRewardInvalidTrace,
    ProcessRewardModelRetired,
    ProcessRewardTraceTooLong,
)

CACHE_HEADER: str = "X-Vlabs-Cache"
CACHE_HEADER_ENABLE_VALUE: str = "enable"
CACHE_TTL_SECONDS: int = 3600
CACHE_KEY_PREFIX: str = "vlabs:prm-score:"

MAX_PROMPT_BYTES: int = 1_000_000
MAX_TRACE_BYTES: int = 4_000_000
"""Larger than Phase 29 (1 MB) — traces are denser per :doc:`PHASE_30_PLAN.md` §8."""


@dataclass(frozen=True)
class HashedPair:
    prompt_hash: str
    trace_hash: str


@dataclass(frozen=True)
class ServeOutcome:
    """Internal per-call outcome used by the route to build the
    response + audit row."""

    score: StubProcessScoreResult
    hashes: HashedPair
    cache_hit: bool
    cache_key: str | None
    latency_ms: int
    segmentation: SegmentationOutcome


def hash_text(text: str) -> str:
    """SHA-256 hex of the UTF-8 bytes of ``text``. Used for both the
    cache key + the audit row's ``prompt_hash``/``trace_hash``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def joined_trace_hash(steps: Sequence[str]) -> str:
    """Hash of the joined-by-newline step list. Stable wrt input
    order so the cache key is deterministic for identical pre-segmented
    input."""
    return hash_text("\n".join(steps))


def cache_key_for(model_id: str, prompt: str, trace_hash: str) -> str:
    """Build the Redis cache key per :doc:`PHASE_30_PLAN.md` §5 D10-B.

    Shape: ``vlabs:prm-score:{model_id}:{sha(prompt)}:{trace_hash}``.
    Plaintext NEVER lands in Redis — only the hashes plus the cached
    response payload.
    """
    return f"{CACHE_KEY_PREFIX}{model_id}:{hash_text(prompt)}:{trace_hash}"


def cache_enabled(headers: Mapping[str, str]) -> bool:
    """Inspect request headers for ``X-Vlabs-Cache: enable`` (D10-B
    opt-in default-off). Both the header name AND the value are
    case-insensitive on the boundary check."""
    for key, value in headers.items():
        if key.lower() == CACHE_HEADER.lower():
            return str(value).strip().lower() == CACHE_HEADER_ENABLE_VALUE
    return False


def validate_inputs(
    prompt: str, reasoning_trace: str | Sequence[str]
) -> None:
    """Reject obviously-broken inputs early."""
    if not prompt.strip():
        raise ProcessRewardInvalidTrace(detail="prompt is empty")
    if isinstance(reasoning_trace, str):
        if not reasoning_trace.strip():
            raise ProcessRewardInvalidTrace(detail="reasoning_trace is empty")
        if len(reasoning_trace.encode("utf-8")) > MAX_TRACE_BYTES:
            raise ProcessRewardInvalidTrace(
                detail=f"reasoning_trace exceeds {MAX_TRACE_BYTES} bytes"
            )
    else:
        if not reasoning_trace:
            raise ProcessRewardInvalidTrace(
                detail="reasoning_trace must be non-empty"
            )
        for i, step in enumerate(reasoning_trace):
            if not isinstance(step, str):
                raise ProcessRewardInvalidTrace(
                    detail=f"reasoning_trace[{i}] is not a string"
                )
        joined = "\n".join(reasoning_trace)
        if len(joined.encode("utf-8")) > MAX_TRACE_BYTES:
            raise ProcessRewardInvalidTrace(
                detail=f"reasoning_trace exceeds {MAX_TRACE_BYTES} bytes"
            )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ProcessRewardInvalidTrace(
            detail=f"prompt exceeds {MAX_PROMPT_BYTES} bytes"
        )


def assert_servable(model: ProcessRewardModel) -> None:
    """Reject calls against retired models with a 410 Gone (D12-B
    lifecycle). ``training`` rows are routed to
    :class:`ProcessRewardModelNotFound` upstream — they're admin-only.
    """
    if model.status == "retired":
        raise ProcessRewardModelRetired(
            detail=f"model_id={model.model_id!r} retired"
        )


async def serve_score(
    *,
    model: ProcessRewardModel,
    prompt: str,
    reasoning_trace: str | Sequence[str],
    cache_get: Any | None = None,
    cache_set: Any | None = None,
    cache_on: bool = False,
    seed: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ServeOutcome:
    """Score a (prompt, trace) pair and emit a :class:`ServeOutcome`.

    The trace is segmented via the D14-D hybrid segmenter; if
    segmentation produces a step count > ``max_steps``, the row is
    flagged ``truncated=True`` (R15) but still served (the segmenter
    returns a truncated ``SegmentationOutcome`` rather than raising).
    Empty / whitespace-only traces raise :class:`ProcessRewardInvalidTrace`.

    ``cache_get``/``cache_set`` mirror Phase 29 — production passes
    Redis-backed coroutines, tests pass an in-memory dict.
    """
    validate_inputs(prompt, reasoning_trace)
    assert_servable(model)

    started = time.perf_counter()
    segmentation = segment_trace(reasoning_trace, max_steps=max_steps)
    if segmentation.step_count == 0:
        raise ProcessRewardInvalidTrace(detail="segmentation produced 0 steps")
    if segmentation.step_count > max_steps:
        # Defence in depth — the segmenter already truncated, but we
        # surface a clean 413 if a future segmenter version forgets.
        raise ProcessRewardTraceTooLong(
            detail=f"step_count={segmentation.step_count} exceeds {max_steps}"
        )

    trace_hash = joined_trace_hash(segmentation.steps)
    hashes = HashedPair(
        prompt_hash=hash_text(prompt), trace_hash=trace_hash
    )
    key = cache_key_for(model.model_id, prompt, trace_hash) if cache_on else None

    cached_payload: StubProcessScoreResult | None = None
    if key is not None and cache_get is not None:
        try:
            raw = await cache_get(key)
        except Exception:  # noqa: BLE001
            raw = None
        if raw:
            try:
                cached_payload = _payload_from_cache(raw, model.model_id)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                cached_payload = None

    if cached_payload is not None:
        latency = int((time.perf_counter() - started) * 1000)
        return ServeOutcome(
            score=cached_payload,
            hashes=hashes,
            cache_hit=True,
            cache_key=key,
            latency_ms=latency,
            segmentation=segmentation,
        )

    score = stub_process_score(
        prompt,
        segmentation.steps,
        model_id=model.model_id,
        seed=seed,
        segmentation_warning=segmentation.warning,
    )

    if key is not None and cache_set is not None:
        with contextlib.suppress(Exception):
            await cache_set(key, _payload_to_cache(score), CACHE_TTL_SECONDS)

    latency = int((time.perf_counter() - started) * 1000)
    return ServeOutcome(
        score=score,
        hashes=hashes,
        cache_hit=False,
        cache_key=key,
        latency_ms=latency,
        segmentation=segmentation,
    )


def _payload_to_cache(score: StubProcessScoreResult) -> str:
    return json.dumps(score.to_dict(), sort_keys=True, ensure_ascii=False)


def _payload_from_cache(raw: Any, model_id: str) -> StubProcessScoreResult:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"cache value must be str/bytes; got {type(raw)!r}")
    payload = json.loads(raw)
    step_rewards = tuple(float(r) for r in payload["step_rewards"])
    step_cis_raw = payload["step_confidence_intervals"]
    step_cis = tuple(
        (float(ci[0]), float(ci[1])) for ci in step_cis_raw
    )
    agg_ci_raw = payload["aggregate_confidence_interval"]
    agg_ci = (float(agg_ci_raw[0]), float(agg_ci_raw[1]))
    return StubProcessScoreResult(
        step_rewards=step_rewards,
        step_confidence_intervals=step_cis,
        aggregate_reward=float(payload["aggregate_reward"]),
        aggregate_confidence_interval=agg_ci,
        coverage_guarantee=float(
            payload.get("coverage_guarantee", DEFAULT_COVERAGE_GUARANTEE)
        ),
        step_count=int(payload.get("step_count", len(step_rewards))),
        model_id=str(payload.get("model_id", model_id)),
        schema_version=str(
            payload.get("schema_version", DEFAULT_SCHEMA_VERSION)
        ),
        cache_hit=False,
        latency_ms=int(payload.get("latency_ms", 0)),
        segmentation_warning=payload.get("segmentation_warning"),
    )


__all__ = [
    "CACHE_HEADER",
    "CACHE_HEADER_ENABLE_VALUE",
    "CACHE_KEY_PREFIX",
    "CACHE_TTL_SECONDS",
    "HashedPair",
    "MAX_PROMPT_BYTES",
    "MAX_TRACE_BYTES",
    "ServeOutcome",
    "assert_servable",
    "cache_enabled",
    "cache_key_for",
    "hash_text",
    "joined_trace_hash",
    "serve_score",
    "validate_inputs",
]
