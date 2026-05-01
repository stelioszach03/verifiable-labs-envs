"""POST /v1/billing/checkout + /v1/billing/portal — Clerk-authed."""
from __future__ import annotations


async def test_checkout_creates_stripe_session(
    client, clerk_user, stub_clerk_verify, stub_stripe
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    r = await client.post(
        "/v1/billing/checkout",
        json={"tier": "pro"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tier"] == "pro"
    assert body["url"].startswith("https://stripe.test/checkout/")
    assert len(stub_stripe["checkouts"]) == 1
    assert stub_stripe["checkouts"][0]["client_reference_id"] == str(user.id)


async def test_checkout_rejects_missing_clerk_token(
    client, stub_stripe
) -> None:
    r = await client.post("/v1/billing/checkout", json={"tier": "pro"})
    assert r.status_code == 401
    assert r.json()["code"] == "invalid_clerk_token"


async def test_checkout_rejects_invalid_tier(
    client, clerk_user, stub_clerk_verify, stub_stripe
) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.post(
        "/v1/billing/checkout",
        json={"tier": "free"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 422  # pydantic Literal violation


async def test_portal_creates_stripe_session(
    client, clerk_user, stub_clerk_verify, stub_stripe
) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.post(
        "/v1/billing/portal",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"].startswith("https://stripe.test/portal/")
    assert len(stub_stripe["portals"]) == 1


async def test_portal_reuses_existing_customer(
    client, clerk_user, stub_clerk_verify, stub_stripe
) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user

    r1 = await client.post(
        "/v1/billing/portal", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    assert r1.status_code == 200
    r2 = await client.post(
        "/v1/billing/portal", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    assert r2.status_code == 200
    # Customer.create called only once across the two portal calls.
    assert len(stub_stripe["customers"]) == 1


# ── Stage C: deferred-billing mode ───────────────────────────────────


async def test_checkout_deferred_mode_returns_503(
    client, clerk_user, stub_clerk_verify, monkeypatch
) -> None:
    """When VLABS_BILLING_ENABLED=false, /v1/billing/checkout 503s — Stripe SDK never touched."""
    monkeypatch.setenv("VLABS_BILLING_ENABLED", "false")
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.post(
        "/v1/billing/checkout",
        json={"tier": "pro"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["code"] == "billing_not_activated"
    assert "free tier" in body["title"].lower()


async def test_portal_deferred_mode_returns_503(
    client, clerk_user, stub_clerk_verify, monkeypatch
) -> None:
    monkeypatch.setenv("VLABS_BILLING_ENABLED", "false")
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.post(
        "/v1/billing/portal",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 503
    assert r.json()["code"] == "billing_not_activated"
