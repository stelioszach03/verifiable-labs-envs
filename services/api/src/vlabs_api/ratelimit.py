"""Tier-aware in-memory sliding-window rate limiter.

Single-instance only. Stage C will swap the storage backend to Redis
(``redis.asyncio``) when we deploy to multi-replica Fly.io machines —
the call-site contract (`enforce_rate_limit` FastAPI dependency) does
not change.

Why custom and not slowapi: slowapi's ``limit_provider`` callable
contract requires either zero args or a single ``key`` arg, which
makes reading per-request tier from ``request.state.auth`` awkward.
A 30-LOC sliding-window over a dict + lock is simpler, tested, and
swappable for Redis later.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import Depends, Request

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.errors import RateLimited
from vlabs_api.usage import tier_limits

if TYPE_CHECKING:
    pass


_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_LOCK = Lock()


def _check_and_increment(key: str, rpm: int) -> tuple[bool, float]:
    """Sliding-window count over the last 60s. Returns ``(allowed, retry_after)``."""
    now = time.monotonic()
    cutoff = now - 60.0
    with _LOCK:
        bucket = _BUCKETS[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= rpm:
            oldest = bucket[0]
            return False, max(0.0, 60.0 - (now - oldest))
        bucket.append(now)
        return True, 0.0


def reset_for_tests() -> None:
    """Clear all buckets — only safe to call from tests."""
    with _LOCK:
        _BUCKETS.clear()


async def enforce_rate_limit(
    request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> AuthContext:
    """FastAPI dependency: gate the request on the tier's RPM cap."""
    _, rpm = tier_limits(auth.tier)
    allowed, retry_after = _check_and_increment(f"key:{auth.api_key_id}", rpm)
    if not allowed:
        raise RateLimited(
            detail=f"tier={auth.tier} rpm={rpm}",
            retry_after=int(retry_after) + 1,
        )
    return auth


__all__ = ["enforce_rate_limit", "reset_for_tests"]
