"""POST /v1/billing/webhook — signature verification, idempotency, event sync."""
from __future__ import annotations

import json
import time
from typing import Any

import pytest
from sqlalchemy import select


def _make_subscription_event(event_id: str, sub_id: str, customer_id: str, status: str = "active") -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": event_id,
        "type": "customer.subscription.created",
        "data": {
            "object": {
                "id": sub_id,
                "customer": customer_id,
                "status": status,
                "current_period_start": now - 60,
                "current_period_end": now + 30 * 86400,
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_test_pro"}}]},
            }
        },
    }


@pytest.fixture
def stub_webhook_verify(monkeypatch):
    """Skip Stripe signature verification — return whatever payload is passed."""
    from vlabs_api import billing

    def _factory(event: dict[str, Any]):
        def _stub(payload: bytes, signature: str):
            return event

        monkeypatch.setattr(billing, "verify_webhook_signature", _stub)
        # webhook.py imports the function directly, so patch there too
        from vlabs_api.routes import webhook as wh

        monkeypatch.setattr(wh, "verify_webhook_signature", _stub)

    return _factory


async def test_webhook_rejects_missing_signature(client, stub_stripe) -> None:
    r = await client.post("/v1/billing/webhook", content=b"{}")
    assert r.status_code == 400
    assert r.json()["code"] == "webhook_signature_invalid"


async def test_webhook_subscription_created_persists_row(
    client, clerk_user, stub_clerk_verify, stub_stripe, stub_webhook_verify, session
) -> None:
    """Full happy path: dashboard creates customer via portal stub, then
    subscription.created webhook syncs the subscription row."""
    from vlabs_api.db import Subscription, User, _SessionFactory

    stub_clerk_verify()
    fake_jwt, user = clerk_user
    # Trigger ensure_stripe_customer via /v1/billing/portal so user has a stripe_customer_id.
    await client.post(
        "/v1/billing/portal", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    # Read in a *fresh* session so we see what the FastAPI request committed.
    async with _SessionFactory() as s:
        res = await s.execute(select(User).where(User.id == user.id))
        customer_id = res.scalar_one().stripe_customer_id
    assert customer_id is not None

    event = _make_subscription_event(
        "evt_test_001", "sub_test_001", customer_id, status="active"
    )
    stub_webhook_verify(event)

    r = await client.post(
        "/v1/billing/webhook",
        content=json.dumps(event).encode(),
        headers={"Stripe-Signature": "stub"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True}

    async with _SessionFactory() as s:
        res = await s.execute(
            select(Subscription).where(Subscription.stripe_subscription_id == "sub_test_001")
        )
        sub = res.scalar_one()
        assert sub.tier == "pro"
        assert sub.status == "active"


async def test_webhook_idempotent_replay(
    client, clerk_user, stub_clerk_verify, stub_stripe, stub_webhook_verify, session
) -> None:
    from vlabs_api.db import StripeEvent, User, _SessionFactory

    stub_clerk_verify()
    fake_jwt, user = clerk_user
    await client.post(
        "/v1/billing/portal", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    async with _SessionFactory() as s:
        res = await s.execute(select(User).where(User.id == user.id))
        customer_id = res.scalar_one().stripe_customer_id

    event = _make_subscription_event("evt_test_replay", "sub_test_replay", customer_id)
    stub_webhook_verify(event)

    r1 = await client.post(
        "/v1/billing/webhook",
        content=json.dumps(event).encode(),
        headers={"Stripe-Signature": "stub"},
    )
    assert r1.status_code == 200

    r2 = await client.post(
        "/v1/billing/webhook",
        content=json.dumps(event).encode(),
        headers={"Stripe-Signature": "stub"},
    )
    assert r2.status_code == 200
    assert r2.json() == {"deduped": True}

    async with _SessionFactory() as s:
        res = await s.execute(
            select(StripeEvent).where(StripeEvent.event_id == "evt_test_replay")
        )
        rows = res.scalars().all()
        assert len(rows) == 1


async def test_webhook_unhandled_event_type_is_200(
    client, stub_stripe, stub_webhook_verify
) -> None:
    event = {
        "id": "evt_test_random",
        "type": "customer.tax_id.created",  # not in _HANDLED_EVENTS
        "data": {"object": {}},
    }
    stub_webhook_verify(event)
    r = await client.post(
        "/v1/billing/webhook",
        content=json.dumps(event).encode(),
        headers={"Stripe-Signature": "stub"},
    )
    assert r.status_code == 200
    assert r.json() == {"ignored": True}


async def test_webhook_deferred_mode_short_circuits(
    client, monkeypatch
) -> None:
    """VLABS_BILLING_ENABLED=false: webhook returns 200 deferred, never instantiates Stripe."""
    monkeypatch.setenv("VLABS_BILLING_ENABLED", "false")
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    r = await client.post(
        "/v1/billing/webhook",
        content=b'{"id":"evt_test_deferred"}',
        headers={"Stripe-Signature": "would-not-even-be-checked"},
    )
    assert r.status_code == 200
    assert r.json() == {"deferred": True}


async def test_webhook_subscription_for_unknown_customer_is_noop(
    client, stub_stripe, stub_webhook_verify, session
) -> None:
    """If the stripe customer hasn't been linked to a user, don't crash."""
    from vlabs_api.db import Subscription

    event = _make_subscription_event("evt_test_orphan", "sub_test_orphan", "cus_unknown")
    stub_webhook_verify(event)
    r = await client.post(
        "/v1/billing/webhook",
        content=json.dumps(event).encode(),
        headers={"Stripe-Signature": "stub"},
    )
    assert r.status_code == 200
    res = await session.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == "sub_test_orphan")
    )
    assert res.scalar_one_or_none() is None
