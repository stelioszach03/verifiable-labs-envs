"""Stripe wrapper — checkout session, billing portal, subscription sync.

All Stripe API calls go through this module so we have one place to
mock in tests and one place to flip from sk_test_ to sk_live_ when the
Delaware C-corp is registered (Stage C-or-later switch).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import stripe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import get_settings
from vlabs_api.db import Subscription, User
from vlabs_api.errors import StripeNotConfigured


def _client() -> Any:
    """Configure the Stripe SDK with the (test-mode) secret key."""
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise StripeNotConfigured(detail="set STRIPE_SECRET_KEY")
    if not settings.stripe_secret_key.startswith("sk_test_"):
        raise StripeNotConfigured(
            detail=(
                "Stage B requires STRIPE_SECRET_KEY to start with 'sk_test_'. "
                "Live-mode keys are blocked until the Delaware C-corp resolves."
            )
        )
    stripe.api_key = settings.stripe_secret_key
    return stripe


def _resolve_price_id(tier: str) -> str:
    settings = get_settings()
    price = {
        "pro": settings.stripe_price_id_pro,
        "team": settings.stripe_price_id_team,
    }.get(tier)
    if not price:
        raise StripeNotConfigured(detail=f"STRIPE_PRICE_ID_{tier.upper()} is unset")
    return price


def _resolve_overage_price_id(tier: str) -> str | None:
    settings = get_settings()
    return {
        "pro": settings.stripe_price_id_pro_overage,
        "team": settings.stripe_price_id_team_overage,
    }.get(tier)


async def ensure_stripe_customer(session: AsyncSession, user: User) -> str:
    """Lazily create a Stripe Customer for the user; return the id."""
    if user.stripe_customer_id:
        return user.stripe_customer_id
    s = _client()
    customer = s.Customer.create(
        email=user.email,
        name=user.name,
        metadata={"vlabs_user_id": str(user.id)},
    )
    user.stripe_customer_id = customer["id"]
    await session.commit()
    await session.refresh(user)
    return customer["id"]


async def create_checkout_session(
    session: AsyncSession,
    user: User,
    tier: str,
) -> str:
    """Return the Stripe Checkout URL for upgrading to ``tier``."""
    if tier not in ("pro", "team"):
        raise StripeNotConfigured(detail=f"unsupported tier {tier!r}")
    s = _client()
    settings = get_settings()
    customer_id = await ensure_stripe_customer(session, user)
    line_items = [{"price": _resolve_price_id(tier), "quantity": 1}]
    overage = _resolve_overage_price_id(tier)
    if overage:
        line_items.append({"price": overage})
    checkout = s.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=line_items,
        success_url=settings.stripe_checkout_success_url,
        cancel_url=settings.stripe_checkout_cancel_url,
        client_reference_id=str(user.id),
        metadata={"tier": tier, "vlabs_user_id": str(user.id)},
    )
    return str(checkout["url"])


async def create_billing_portal_session(
    session: AsyncSession, user: User
) -> str:
    """Return the Stripe Billing Portal URL for self-service management."""
    s = _client()
    settings = get_settings()
    customer_id = await ensure_stripe_customer(session, user)
    portal = s.billing_portal.Session.create(
        customer=customer_id,
        return_url=settings.stripe_billing_portal_return_url,
    )
    return str(portal["url"])


def verify_webhook_signature(payload: bytes, signature: str) -> stripe.Event:
    """Verify a Stripe webhook signature; return the parsed event.

    Raises :class:`stripe.SignatureVerificationError` on bad sigs —
    caller wraps that into an :class:`WebhookSignatureInvalid`.
    """
    settings = get_settings()
    if not settings.stripe_webhook_secret:
        raise StripeNotConfigured(detail="set STRIPE_WEBHOOK_SECRET")
    return stripe.Webhook.construct_event(
        payload, signature, settings.stripe_webhook_secret
    )


async def sync_subscription_from_event(
    session: AsyncSession,
    sub_obj: dict[str, Any],
) -> Subscription | None:
    """Upsert a ``subscriptions`` row from a Stripe Subscription object."""
    stripe_id = sub_obj["id"]
    customer_id = sub_obj["customer"]

    # Find the user via stripe_customer_id; if missing, abort silently —
    # we'll get the next event after the user is linked.
    user_q = await session.execute(
        select(User).where(User.stripe_customer_id == customer_id)
    )
    user = user_q.scalar_one_or_none()
    if user is None:
        return None

    items = sub_obj.get("items", {}).get("data", [])
    tier = "pro"
    if items:
        first_price = items[0].get("price", {}).get("id")
        settings = get_settings()
        if first_price and first_price == settings.stripe_price_id_team:
            tier = "team"

    period_start = datetime.fromtimestamp(
        sub_obj["current_period_start"], tz=UTC
    )
    period_end = datetime.fromtimestamp(
        sub_obj["current_period_end"], tz=UTC
    )

    existing_q = await session.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_id)
    )
    existing = existing_q.scalar_one_or_none()
    if existing is None:
        sub = Subscription(
            user_id=user.id,
            stripe_subscription_id=stripe_id,
            tier=tier,
            status=sub_obj["status"],
            current_period_start=period_start,
            current_period_end=period_end,
            cancel_at_period_end=bool(sub_obj.get("cancel_at_period_end", False)),
        )
        session.add(sub)
    else:
        existing.tier = tier
        existing.status = sub_obj["status"]
        existing.current_period_start = period_start
        existing.current_period_end = period_end
        existing.cancel_at_period_end = bool(
            sub_obj.get("cancel_at_period_end", False)
        )
        existing.updated_at = datetime.now(UTC)
        sub = existing
    await session.commit()
    await session.refresh(sub)
    return sub


__all__ = [
    "ensure_stripe_customer",
    "create_checkout_session",
    "create_billing_portal_session",
    "verify_webhook_signature",
    "sync_subscription_from_event",
]
