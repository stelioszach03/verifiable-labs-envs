"""``/v1/keys/*`` — issue, list, revoke API keys via Clerk auth.

These are management-plane endpoints (Clerk JWT, not X-Vlabs-Key) so a
dashboard user can mint API keys for their own account without leaking
their Clerk session into the data plane.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import (
    generate_plaintext_key,
    hash_plaintext_key,
    key_prefix,
)
from vlabs_api.clerk_auth import require_clerk_user
from vlabs_api.db import APIKey, User, get_db
from vlabs_api.errors import APIKeyNotFoundForUser
from vlabs_api.schemas import (
    APIKeyCreated,
    APIKeyInfo,
    APIKeyList,
    CreateAPIKeyRequest,
)

router = APIRouter(tags=["management"])


@router.post("/keys", response_model=APIKeyCreated)
async def create_api_key(
    payload: CreateAPIKeyRequest,
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> APIKeyCreated:
    plaintext = generate_plaintext_key()
    row = APIKey(
        user_id=user.id,
        key_hash=hash_plaintext_key(plaintext),
        key_prefix=key_prefix(plaintext),
        name=payload.name,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return APIKeyCreated(
        id=str(row.id),
        prefix=row.key_prefix,
        name=row.name,
        created_at=row.created_at,
        plaintext_key=plaintext,  # only returned this once
    )


@router.get("/keys", response_model=APIKeyList)
async def list_api_keys(
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> APIKeyList:
    res = await session.execute(
        select(APIKey)
        .where(APIKey.user_id == user.id)
        .order_by(APIKey.created_at.desc())
    )
    rows = res.scalars().all()
    return APIKeyList(
        items=[
            APIKeyInfo(
                id=str(r.id),
                prefix=r.key_prefix,
                name=r.name,
                created_at=r.created_at,
                last_used_at=r.last_used_at,
                revoked_at=r.revoked_at,
            )
            for r in rows
        ]
    )


@router.delete("/keys/{key_id}", response_model=APIKeyInfo)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> APIKeyInfo:
    try:
        uid = uuid.UUID(key_id)
    except ValueError as exc:
        raise APIKeyNotFoundForUser(detail=key_id) from exc
    res = await session.execute(
        select(APIKey).where(APIKey.id == uid, APIKey.user_id == user.id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise APIKeyNotFoundForUser(detail=key_id)
    row.revoked_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(row)
    return APIKeyInfo(
        id=str(row.id),
        prefix=row.key_prefix,
        name=row.name,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
    )
