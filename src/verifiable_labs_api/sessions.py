"""In-memory session store with TTL eviction.

Each session caches the ``Instance`` object the env produced (so the
score call doesn't have to regenerate it), the env handle, and a
list of past submissions. Sessions expire after ``ttl_seconds``;
expired entries are evicted lazily on every ``get``/``put`` call.

Tier-1 alpha is single-process; for multi-replica deploys this would
be replaced with Redis. The interface here mirrors a Redis client so
that swap is mechanical.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Submission:
    answer_text: str | None
    answer: dict[str, Any] | None
    reward: float
    components: dict[str, float]
    coverage: float | None
    parse_ok: bool
    meta: dict[str, Any]
    submitted_at: datetime


@dataclass
class Session:
    """One evaluation session, cached on the server."""

    session_id: str
    env_id: str
    seed: int
    instance: Any  # the env's Instance dataclass — opaque to this module
    env: Any  # the env handle — opaque to this module
    created_at: datetime
    expires_at: datetime
    submissions: list[Submission] = field(default_factory=list)
    complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """Thread-safe in-memory session store.

    Public ops: ``put``, ``get``, ``len``, ``evict_expired``. ``get``
    raises ``KeyError`` for missing/expired sessions; the caller maps
    that to HTTP 404.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0; got {ttl_seconds}")
        self._ttl = int(ttl_seconds)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._created_at = time.monotonic()

    def make_session(
        self,
        env_id: str,
        seed: int,
        instance: Any,
        env: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        sid = str(uuid.uuid4())
        now = _utcnow()
        session = Session(
            session_id=sid,
            env_id=env_id,
            seed=seed,
            instance=instance,
            env=env,
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl),
            metadata=metadata or {},
        )
        with self._lock:
            self.evict_expired_locked()
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            self.evict_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return session

    def evict_expired_locked(self) -> int:
        """Remove expired sessions. Caller must hold ``self._lock``.

        Returns the number of evicted sessions.
        """
        now = _utcnow()
        expired = [sid for sid, s in self._sessions.items() if s.expires_at <= now]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def evict_expired(self) -> int:
        with self._lock:
            return self.evict_expired_locked()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._created_at

    def clear(self) -> None:
        """Test helper — wipe all sessions."""
        with self._lock:
            self._sessions.clear()
