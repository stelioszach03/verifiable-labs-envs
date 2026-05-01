"""``POST /v1/billing/checkout`` and ``POST /v1/billing/portal`` — Clerk auth.

While ``VLABS_BILLING_ENABLED=false`` (default in Stage C until the
Delaware C-corp registration completes), both endpoints return
``503 billing_not_activated`` and the Stripe SDK is never instantiated.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.billing import (
    create_billing_portal_session,
    create_checkout_session,
)
from vlabs_api.clerk_auth import require_clerk_user
from vlabs_api.config import get_settings
from vlabs_api.db import User, get_db
from vlabs_api.errors import BillingNotActivated
from vlabs_api.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    PortalResponse,
)

router = APIRouter(tags=["billing"])


def _require_billing_enabled() -> None:
    if not get_settings().vlabs_billing_enabled:
        raise BillingNotActivated()


@router.post("/billing/checkout", response_model=CheckoutResponse)
async def billing_checkout(
    payload: CheckoutRequest,
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> CheckoutResponse:
    _require_billing_enabled()
    url = await create_checkout_session(session, user, payload.tier)
    return CheckoutResponse(url=url, tier=payload.tier)


@router.post("/billing/portal", response_model=PortalResponse)
async def billing_portal(
    user: User = Depends(require_clerk_user),
    session: AsyncSession = Depends(get_db),
) -> PortalResponse:
    _require_billing_enabled()
    url = await create_billing_portal_session(session, user)
    return PortalResponse(url=url)
