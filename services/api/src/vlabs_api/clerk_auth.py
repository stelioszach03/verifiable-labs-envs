"""Clerk JWT verification for the management plane.

Used by `/v1/billing/*` and `/v1/keys/*` — endpoints the dashboard
hits, NOT the data plane. The data plane (`/v1/calibrate`,
`/v1/predict`, etc.) only accepts ``X-Vlabs-Key``.

Verification flow:
  1. Read ``Authorization: Bearer <jwt>`` header.
  2. Fetch JWKS from Clerk (cached for the process lifetime).
  3. Verify signature with the matching key from the JWKS.
  4. Look up or create the matching ``users`` row by ``clerk_user_id``.
"""
from __future__ import annotations

import threading
from typing import Any

import jwt
from fastapi import Depends, Request
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import get_settings
from vlabs_api.db import User, get_db
from vlabs_api.errors import ClerkNotConfigured, InvalidClerkToken

_jwks_client: PyJWKClient | None = None
_jwks_lock = threading.Lock()


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client
    with _jwks_lock:
        if _jwks_client is not None:
            return _jwks_client
        settings = get_settings()
        url = settings.clerk_jwks_url
        if url is None:
            issuer = settings.clerk_jwt_issuer
            if not issuer:
                raise ClerkNotConfigured(
                    detail="set CLERK_JWT_ISSUER or CLERK_JWKS_URL"
                )
            url = issuer.rstrip("/") + "/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, cache_keys=True, lifespan=3600)
        return _jwks_client


def _read_bearer(request: Request) -> str:
    auth_header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise InvalidClerkToken(detail="missing or malformed Authorization header")
    return auth_header.split(" ", 1)[1].strip()


def _verify_jwt(token: str) -> dict[str, Any]:
    settings = get_settings()
    issuer = settings.clerk_jwt_issuer
    if not issuer:
        raise ClerkNotConfigured(detail="CLERK_JWT_ISSUER is required")
    try:
        client = _get_jwks_client()
        signing_key = client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"require": ["sub", "exp", "iat"]},
        )
        return claims
    except jwt.PyJWTError as exc:
        raise InvalidClerkToken(detail=str(exc)) from exc


async def _resolve_user(session: AsyncSession, claims: dict[str, Any]) -> User:
    clerk_id = claims.get("sub")
    if not clerk_id:
        raise InvalidClerkToken(detail="JWT has no 'sub' claim")
    res = await session.execute(select(User).where(User.clerk_user_id == clerk_id))
    user = res.scalar_one_or_none()
    if user is not None:
        return user

    email = claims.get("email")
    if not email:
        # Clerk JWT templates can be configured to include email; fall back
        # to a stable synthetic placeholder so the user row exists. The
        # email gets backfilled by the next webhook or dashboard form.
        email = f"{clerk_id}@clerk.placeholder"
    user = User(
        email=email,
        name=claims.get("name") or claims.get("first_name"),
        clerk_user_id=clerk_id,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def require_clerk_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency: verify Clerk JWT, return the matching User row."""
    token = _read_bearer(request)
    claims = _verify_jwt(token)
    user = await _resolve_user(session, claims)
    request.state.clerk_user = user
    return user


__all__ = ["require_clerk_user"]
