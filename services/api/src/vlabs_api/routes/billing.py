"""``POST /v1/billing/checkout`` and ``POST /v1/billing/portal`` — Clerk auth."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.billing import (
    create_billing_portal_session,
    create_checkout_session,
)
from vlabs_api.clerk_auth import require_clerk_user
from vlabs_api.db import User, get_db
from vlabs_api.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    PortalResponse,
)

router = APIRouter(tags=["billing"])


@router.post("/billing/checkout", response_model=CheckoutResponse)
async def billing_checkout(
    payload: CheckoutRequest,
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> CheckoutResponse:
    url = await create_checkout_session(session, user, payload.tier)
    return CheckoutResponse(url=url, tier=payload.tier)


@router.post("/billing/portal", response_model=PortalResponse)
async def billing_portal(
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> PortalResponse:
    url = await create_billing_portal_session(session, user)
    return PortalResponse(url=url)
