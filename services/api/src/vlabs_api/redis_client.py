"""Minimal async client for Upstash Redis REST API.

Used only by the rate-limit module. Plain ``httpx.AsyncClient`` against
the Upstash REST endpoint — no upstash-redis SDK dep, keeps the
service's dependency tree tight.

The REST API accepts each Redis command as a JSON-encoded list:
``["ZADD", key, score, member]`` — and pipelines as a list of those.
We only use a tiny subset (ZADD, ZCARD, ZREMRANGEBYSCORE, ZRANGE,
EXPIRE, DEL).
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog

from vlabs_api.config import get_settings

log = structlog.get_logger(__name__)


class UpstashClient:
    """Thread-safe-ish async wrapper around an Upstash REST endpoint."""

    def __init__(self, url: str, token: str, timeout: float = 5.0) -> None:
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = httpx.AsyncClient(timeout=timeout, headers=self._headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def pipeline(self, *commands: list[Any]) -> list[Any]:
        """Send N commands in one round-trip; return list of results."""
        body = [list(map(str, cmd)) for cmd in commands]
        resp = await self._client.post(f"{self._url}/pipeline", json=body)
        resp.raise_for_status()
        data = resp.json()
        # Each entry is {"result": ...} or {"error": "..."}
        out: list[Any] = []
        for entry in data:
            if isinstance(entry, dict) and "error" in entry:
                raise RuntimeError(f"upstash error: {entry['error']}")
            out.append(entry["result"] if isinstance(entry, dict) else entry)
        return out

    async def ping(self) -> bool:
        resp = await self._client.get(f"{self._url}/PING")
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") == "PONG"


_client: UpstashClient | None = None


def get_client() -> UpstashClient | None:
    """Lazy singleton — returns None when Upstash is not configured."""
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if not s.upstash_redis_rest_url or not s.upstash_redis_rest_token:
        return None
    _client = UpstashClient(s.upstash_redis_rest_url, s.upstash_redis_rest_token)
    log.info("redis_client.initialised", host=s.upstash_redis_rest_url.split("//", 1)[-1])
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


__all__ = ["UpstashClient", "get_client", "aclose"]
