"""Service layer for the distilled reward model endpoints (Phase 29.E).

Wraps :func:`verifiable_labs_envs.reward_distillation.stub_inference.stub_score`
in the API surface — adds prompt/response hashing (D11-C privacy
posture), Redis cache key derivation, and audit-row construction. The
trained-student inference layer arrives in 29.G; until then every
score returns the canonical stub payload with
``schema_version="v0.1.0-stub"``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verifiable_labs_envs.reward_distillation.stub_inference import (
    DEFAULT_COVERAGE_GUARANTEE,
    DEFAULT_SCHEMA_VERSION,
    StubScoreResult,
    stub_score,
)

from vlabs_api.db import RewardModel
from vlabs_api.errors import RewardModelInvalidRequest, RewardModelRetired

CACHE_HEADER: str = "X-Vlabs-Cache"
CACHE_HEADER_ENABLE_VALUE: str = "enable"
CACHE_TTL_SECONDS: int = 3600
CACHE_KEY_PREFIX: str = "vlabs:reward-score:"

MAX_PROMPT_BYTES: int = 1_000_000  # 1 MB cap mirrors /v1/score
MAX_RESPONSE_BYTES: int = 1_000_000


@dataclass(frozen=True)
class HashedPair:
    prompt_hash: str
    response_hash: str


@dataclass(frozen=True)
class ServeOutcome:
    """Internal per-call outcome used by the route to build the
    response + audit row."""

    score: StubScoreResult
    hashes: HashedPair
    cache_hit: bool
    cache_key: str | None
    latency_ms: int


def hash_prompt(text: str) -> str:
    """SHA-256 hex of the UTF-8 bytes of ``text``. Used for both the
    cache key + the audit row's ``prompt_hash``/``response_hash``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key_for(model_id: str, prompt: str, response: str) -> str:
    """Build the Redis cache key per :doc:`PHASE_29_PLAN.md` §5 D11-C.

    Shape: ``vlabs:reward-score:{model_id}:{sha(prompt)}:{sha(response)}``.
    Plaintext NEVER lands in Redis — only the hashes plus the cached
    response payload.
    """
    return (
        f"{CACHE_KEY_PREFIX}{model_id}:"
        f"{hash_prompt(prompt)}:{hash_prompt(response)}"
    )


def cache_enabled(headers: Mapping[str, str]) -> bool:
    """Inspect request headers for ``X-Vlabs-Cache: enable`` (D11-C
    opt-in default-off). Both the header name AND the value are
    case-insensitive on the boundary check.
    """
    for key, value in headers.items():
        if key.lower() == CACHE_HEADER.lower():
            return str(value).strip().lower() == CACHE_HEADER_ENABLE_VALUE
    return False


def validate_inputs(prompt: str, response: str) -> None:
    """Reject obviously-broken inputs early.

    Mirrors the Phase 22 ``/v1/score`` posture: 1 MB cap on each side,
    non-empty after stripping.
    """
    if not prompt.strip():
        raise RewardModelInvalidRequest(detail="prompt is empty")
    if not response.strip():
        raise RewardModelInvalidRequest(detail="response is empty")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise RewardModelInvalidRequest(
            detail=f"prompt exceeds {MAX_PROMPT_BYTES} bytes"
        )
    if len(response.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise RewardModelInvalidRequest(
            detail=f"response exceeds {MAX_RESPONSE_BYTES} bytes"
        )


def assert_servable(model: RewardModel) -> None:
    """Reject calls against retired models with a 410 Gone (D12-B
    lifecycle). ``training`` rows are routed to ``RewardModelNotFound``
    upstream — they're admin-only.
    """
    if model.status == "retired":
        raise RewardModelRetired(
            detail=f"model_id={model.model_id!r} retired"
        )


async def serve_score(
    *,
    model: RewardModel,
    prompt: str,
    response: str,
    cache_get: Any | None = None,
    cache_set: Any | None = None,
    cache_on: bool = False,
    seed: int | None = None,
) -> ServeOutcome:
    """Score a (prompt, response) pair and emit a :class:`ServeOutcome`.

    ``cache_get``/``cache_set`` are pluggable so tests pass an
    in-memory dict and production passes Redis-backed coroutines
    matching ``async def get(key) -> str | None`` and ``async def
    set(key, value, ttl_seconds) -> None``.

    Privacy posture (§5 D11-C): the cache key uses SHA-256 hashes of
    the input bytes; plaintext is never persisted in Redis. The cached
    payload is the JSON-encoded :class:`StubScoreResult` (which itself
    carries no plaintext beyond the stub schema sentinel).
    """
    validate_inputs(prompt, response)
    assert_servable(model)

    started = time.perf_counter()
    hashes = HashedPair(
        prompt_hash=hash_prompt(prompt),
        response_hash=hash_prompt(response),
    )
    key = cache_key_for(model.model_id, prompt, response) if cache_on else None

    cached_payload: StubScoreResult | None = None
    if key is not None and cache_get is not None:
        try:
            raw = await cache_get(key)
        except Exception:  # noqa: BLE001 — cache-miss safety
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
        )

    score = stub_score(prompt, response, model_id=model.model_id, seed=seed)

    if key is not None and cache_set is not None:
        # Cache writes never fail the call (Redis flake or fallthrough
        # serialization issue) — :func:`contextlib.suppress` keeps the
        # path clean while still letting cancellation propagate.
        with contextlib.suppress(Exception):
            await cache_set(key, _payload_to_cache(score), CACHE_TTL_SECONDS)

    latency = int((time.perf_counter() - started) * 1000)
    return ServeOutcome(
        score=score,
        hashes=hashes,
        cache_hit=False,
        cache_key=key,
        latency_ms=latency,
    )


def _payload_to_cache(score: StubScoreResult) -> str:
    return json.dumps(score.to_dict(), sort_keys=True, ensure_ascii=False)


def _payload_from_cache(raw: Any, model_id: str) -> StubScoreResult:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"cache value must be str/bytes; got {type(raw)!r}")
    payload = json.loads(raw)
    return StubScoreResult(
        reward=float(payload["reward"]),
        confidence_interval=(
            float(payload["confidence_interval"][0]),
            float(payload["confidence_interval"][1]),
        ),
        coverage_guarantee=float(
            payload.get("coverage_guarantee", DEFAULT_COVERAGE_GUARANTEE)
        ),
        model_id=str(payload.get("model_id", model_id)),
        schema_version=str(
            payload.get("schema_version", DEFAULT_SCHEMA_VERSION)
        ),
        cache_hit=False,  # forced False — caller sets cache_hit=True
        latency_ms=int(payload.get("latency_ms", 0)),
    )


def reward_run_id() -> uuid.UUID:
    """Fresh UUID for the ``reward_model_runs.id`` column. Wrapped in a
    helper so tests can monkeypatch it without touching the ORM."""
    return uuid.uuid4()


__all__ = [
    "CACHE_HEADER",
    "CACHE_HEADER_ENABLE_VALUE",
    "CACHE_KEY_PREFIX",
    "CACHE_TTL_SECONDS",
    "HashedPair",
    "MAX_PROMPT_BYTES",
    "MAX_RESPONSE_BYTES",
    "ServeOutcome",
    "assert_servable",
    "cache_enabled",
    "cache_key_for",
    "hash_prompt",
    "reward_run_id",
    "serve_score",
    "validate_inputs",
]
