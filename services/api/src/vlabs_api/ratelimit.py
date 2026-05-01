"""Tier-aware sliding-window rate limiter.

Two backends, swapped automatically based on env config:

  * **memory** (default): in-process ``deque`` per key. Single-instance
    only — fine for local dev and the tests.
  * **redis**: Upstash REST sliding window via ZSET. Multi-instance
    safe; used in production once both ``UPSTASH_REDIS_REST_URL`` and
    ``UPSTASH_REDIS_REST_TOKEN`` are set.

The call-site contract — ``enforce_rate_limit`` FastAPI dependency —
is identical across backends so route code never changes.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from threading import Lock
from typing import TYPE_CHECKING

import structlog
from fastapi import Depends, Request

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.errors import RateLimited
from vlabs_api.redis_client import get_client as get_redis
from vlabs_api.usage import tier_limits

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

# ── Memory backend (default; used by tests + local dev) ──────────────


_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = Lock()


def _check_and_increment_memory(key: str, rpm: int) -> tuple[bool, float]:
    now = time.monotonic()
    cutoff = now - 60.0
    with _LOCK:
        bucket = _BUCKETS[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= rpm:
            return False, max(0.0, 60.0 - (now - bucket[0]))
        bucket.append(now)
        return True, 0.0


def reset_for_tests() -> None:
    """Clear the in-memory buckets — only safe to call from tests."""
    with _LOCK:
        _BUCKETS.clear()


# ── Redis backend (Upstash REST sliding window over ZSET) ────────────


async def _check_and_increment_redis(key: str, rpm: int) -> tuple[bool, float]:
    """Sliding-window rate limit using ``ZADD`` / ``ZCARD`` /
    ``ZREMRANGEBYSCORE``.

    Race window: between the count check and the ``ZADD`` two requests
    can both succeed when the bucket is at ``rpm − 1``, ending with
    ``rpm + 1`` entries. Acceptable for now; a Lua script atomicity
    upgrade is a follow-up if abuse appears in monitoring.
    """
    client = get_redis()
    if client is None:  # pragma: no cover — guarded at call site
        return True, 0.0
    redis_key = f"vlabs_rl:{key}"
    now_ms = int(time.time() * 1000)
    cutoff = now_ms - 60_000
    try:
        # Step 1: drop expired + count current
        results = await client.pipeline(
            ["ZREMRANGEBYSCORE", redis_key, "-inf", str(cutoff)],
            ["ZCARD", redis_key],
        )
        count = int(results[1])
        if count >= rpm:
            # Look up the oldest entry's score to give an accurate retry-after.
            oldest = await client.pipeline(
                ["ZRANGE", redis_key, "0", "0", "WITHSCORES"]
            )
            try:
                oldest_score = int(oldest[0][1]) if oldest and oldest[0] else cutoff
            except (IndexError, TypeError, ValueError):
                oldest_score = cutoff
            retry_after = max(0.0, 60.0 - (now_ms - oldest_score) / 1000.0)
            return False, retry_after
        # Step 2: insert this request + bump TTL
        member = f"{now_ms}-{uuid.uuid4().hex[:8]}"
        await client.pipeline(
            ["ZADD", redis_key, str(now_ms), member],
            ["EXPIRE", redis_key, "65"],
        )
        return True, 0.0
    except Exception as exc:  # noqa: BLE001
        # Open the gate on Redis failure rather than dropping all traffic.
        # Sentry catches the exception; the API stays available.
        log.warning("ratelimit.redis_error_open", err=type(exc).__name__)
        return True, 0.0


# ── Backend dispatcher ───────────────────────────────────────────────


def _backend() -> str:
    return "redis" if get_redis() is not None else "memory"


async def _check_and_increment(key: str, rpm: int) -> tuple[bool, float]:
    if _backend() == "redis":
        return await _check_and_increment_redis(key, rpm)
    return _check_and_increment_memory(key, rpm)


# ── FastAPI dependency ──────────────────────────────────────────────


async def enforce_rate_limit(
    request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> AuthContext:
    """Gate the request on the tier's RPM cap; raise ``RateLimited`` on excess."""
    _, rpm = tier_limits(auth.tier)
    allowed, retry_after = await _check_and_increment(f"key:{auth.api_key_id}", rpm)
    if not allowed:
        raise RateLimited(
            detail=f"tier={auth.tier} rpm={rpm}",
            retry_after=int(retry_after) + 1,
        )
    return auth


__all__ = ["enforce_rate_limit", "reset_for_tests"]
