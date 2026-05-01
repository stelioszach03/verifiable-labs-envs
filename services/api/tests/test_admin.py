"""GET /v1/admin/dashboard — Clerk auth + allowlist + aggregate stats."""
from __future__ import annotations

import uuid


async def test_admin_rejects_non_admin(client, clerk_user, stub_clerk_verify) -> None:
    """Clerk user with valid JWT but not in allowlist gets 403."""
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.get(
        "/v1/admin/dashboard",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 403
    assert r.json()["code"] == "not_admin"


async def test_admin_rejects_missing_clerk_token(client) -> None:
    r = await client.get("/v1/admin/dashboard")
    assert r.status_code == 401
    assert r.json()["code"] == "invalid_clerk_token"


async def test_admin_dashboard_for_allowlisted_user(
    client, clerk_user, stub_clerk_verify, monkeypatch
) -> None:
    """Same Clerk user lifted into the allowlist returns 200 + counts."""
    stub_clerk_verify()
    fake_jwt, user = clerk_user

    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    r = await client.get(
        "/v1/admin/dashboard",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "counts" in body
    assert body["counts"]["users"] >= 1  # at least the test user
    assert body["counts"]["api_keys_active"] == 0
    assert isinstance(body["most_recent_calibrations"], list)
    assert "billing_enabled" in body


async def test_admin_dashboard_supports_multiple_admins(
    client, clerk_user, stub_clerk_verify, monkeypatch, session
) -> None:
    """Comma-separated allowlist with whitespace tolerated."""
    from vlabs_api.config import get_settings
    from vlabs_api.db import User

    stub_clerk_verify()
    fake_jwt, user = clerk_user

    other = User(
        email=f"other-{uuid.uuid4().hex[:8]}@example.com",
        clerk_user_id="user_other_admin",
    )
    session.add(other)
    await session.commit()

    monkeypatch.setenv(
        "VLABS_ADMIN_CLERK_IDS",
        f"  user_other_admin , {user.clerk_user_id}  ,  user_third  ",
    )
    get_settings.cache_clear()

    r = await client.get(
        "/v1/admin/dashboard",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
