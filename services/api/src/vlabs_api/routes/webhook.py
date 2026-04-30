"""``POST /v1/billing/webhook`` — Stripe event ingest with signature verification + idempotency."""
from __future__ import annotations

from datetime import UTC, datetime

import stripe
import structlog
from fastapi import APIRouter, Depends, Header, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.billing import sync_subscription_from_event, verify_webhook_signature
from vlabs_api.db import StripeEvent, get_db
from vlabs_api.errors import WebhookSignatureInvalid

router = APIRouter(tags=["billing"])
log = structlog.get_logger(__name__)

_HANDLED_EVENTS = {
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.paid",
    "invoice.payment_failed",
    "checkout.session.completed",
}


@router.post("/billing/webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    session: AsyncSession = Depends(get_db),
) -> Response:
    payload = await request.body()
    if not stripe_signature:
        raise WebhookSignatureInvalid(detail="Stripe-Signature header missing")

    try:
        event = verify_webhook_signature(payload, stripe_signature)
    except stripe.SignatureVerificationError as exc:
        raise WebhookSignatureInvalid(detail=str(exc)) from exc

    event_id = event["id"]
    event_type = event["type"]

    # Idempotency check — INSERT first, then process.
    dedup = StripeEvent(event_id=event_id, event_type=event_type)
    session.add(dedup)
    try:
        await session.flush()
    except Exception:
        await session.rollback()
        log.info("webhook.duplicate_replay", event_id=event_id, event_type=event_type)
        return Response(status_code=200, content=b'{"deduped": true}')

    if event_type not in _HANDLED_EVENTS:
        await session.commit()
        log.info(
            "webhook.unhandled_event_type", event_id=event_id, event_type=event_type
        )
        return Response(status_code=200, content=b'{"ignored": true}')

    obj = event["data"]["object"]
    try:
        if event_type.startswith("customer.subscription."):
            await sync_subscription_from_event(session, obj)
        elif event_type == "checkout.session.completed":
            sub_id = obj.get("subscription")
            if sub_id:
                stripe.api_key = stripe.api_key or ""
                try:
                    sub = stripe.Subscription.retrieve(sub_id)
                    await sync_subscription_from_event(session, sub)
                except Exception as exc:
                    log.warning(
                        "webhook.checkout_subscription_fetch_failed",
                        sub_id=sub_id,
                        err=str(exc),
                    )
        # invoice.paid / invoice.payment_failed: no DB mutations beyond
        # the dedup row in Stage B; full metered-billing reconciliation
        # is a Stage C task.

        dedup.processed_at = datetime.now(UTC)
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        dedup.error = str(exc)[:1000]
        await session.commit()
        log.exception("webhook.processing_failed", event_id=event_id)
        # Stripe expects 2xx on receipt; we don't 5xx because that triggers
        # retry storms. Errors land in stripe_events.error for postmortems.
        return Response(status_code=200, content=b'{"ok": false}')

    return Response(status_code=200, content=b'{"ok": true}')
