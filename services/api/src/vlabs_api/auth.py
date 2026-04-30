"""``X-Vlabs-Key`` authentication + per-request context."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import get_settings
from vlabs_api.db import APIKey, get_db
from vlabs_api.errors import InvalidAPIKey
from vlabs_api.usage import Tier, resolve_tier

if TYPE_CHECKING:
    pass

KEY_PREFIX = "vlk_"
KEY_RAW_BYTES = 24  # 32 chars URL-safe base64 after token_urlsafe


@dataclass(frozen=True)
class AuthContext:
    """Attached to ``request.state.auth`` after successful authentication."""

    user_id: uuid.UUID
    api_key_id: uuid.UUID
    tier: Tier


def generate_plaintext_key() -> str:
    """Return a fresh ``vlk_<32-char>`` plaintext API key."""
    return KEY_PREFIX + secrets.token_urlsafe(KEY_RAW_BYTES)[:32]


def hash_plaintext_key(plaintext: str) -> bytes:
    """Compute the peppered SHA-256 hash of a plaintext key."""
    pepper = get_settings().vlabs_api_key_hash_pepper
    h = hashlib.sha256()
    h.update(plaintext.encode("utf-8"))
    h.update(b"|")
    h.update(pepper.encode("utf-8"))
    return h.digest()


def key_prefix(plaintext: str) -> str:
    """Return the first 8 chars of a plaintext key (for UI display)."""
    return plaintext[:8]


def _read_header(request: Request) -> str:
    raw = request.headers.get("x-vlabs-key") or request.headers.get("X-Vlabs-Key")
    if not raw or not raw.startswith(KEY_PREFIX):
        raise InvalidAPIKey(detail="missing or malformed X-Vlabs-Key header")
    return raw


async def require_api_key(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> AuthContext:
    """FastAPI dependency: validate header, attach :class:`AuthContext`."""
    plaintext = _read_header(request)
    key_hash = hash_plaintext_key(plaintext)
    res = await session.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    row = res.scalar_one_or_none()
    if row is None or row.revoked_at is not None:
        raise InvalidAPIKey(detail="API key not found or revoked")

    # Async update of last_used_at — fire-and-forget on commit boundary.
    row.last_used_at = datetime.now(UTC)

    tier = await resolve_tier(session, row.user_id)
    ctx = AuthContext(user_id=row.user_id, api_key_id=row.id, tier=tier)
    request.state.auth = ctx
    return ctx


__all__ = [
    "AuthContext",
    "KEY_PREFIX",
    "generate_plaintext_key",
    "hash_plaintext_key",
    "key_prefix",
    "require_api_key",
]
