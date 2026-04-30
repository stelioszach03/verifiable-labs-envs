"""POST/GET/DELETE /v1/keys/* — Clerk-authed key management."""
from __future__ import annotations

from sqlalchemy import select


async def test_create_returns_plaintext_once(client, clerk_user, stub_clerk_verify) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.post(
        "/v1/keys",
        json={"name": "ci-key"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["plaintext_key"].startswith("vlk_")
    assert len(out["plaintext_key"]) >= 16
    assert out["prefix"] == out["plaintext_key"][:8]
    assert out["name"] == "ci-key"
    assert out["revoked_at"] is None


async def test_list_keys(client, clerk_user, stub_clerk_verify) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    for i in range(3):
        r = await client.post(
            "/v1/keys",
            json={"name": f"key-{i}"},
            headers={"Authorization": f"Bearer {fake_jwt}"},
        )
        assert r.status_code == 200

    r = await client.get(
        "/v1/keys", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 3
    assert "plaintext_key" not in items[0]


async def test_revoke_key(client, clerk_user, stub_clerk_verify, session) -> None:
    from vlabs_api.db import APIKey

    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    create = await client.post(
        "/v1/keys",
        json={"name": "to-revoke"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert create.status_code == 200
    key_id = create.json()["id"]

    delete = await client.delete(
        f"/v1/keys/{key_id}",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert delete.status_code == 200
    assert delete.json()["revoked_at"] is not None

    res = await session.execute(select(APIKey))
    rows = res.scalars().all()
    assert len(rows) == 1
    assert rows[0].revoked_at is not None


async def test_revoke_unknown_key_404(client, clerk_user, stub_clerk_verify) -> None:
    stub_clerk_verify()
    fake_jwt, _ = clerk_user
    r = await client.delete(
        "/v1/keys/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "api_key_not_found"


async def test_keys_isolation_between_clerk_users(
    client, clerk_user, stub_clerk_verify, session
) -> None:
    """A Clerk user only sees their own keys, not other users'."""
    from vlabs_api.db import User

    stub_clerk_verify()
    fake_jwt, user_a = clerk_user

    # Create key as user A
    create = await client.post(
        "/v1/keys",
        json={"name": "user-a-key"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert create.status_code == 200

    # Create user B + their JWT
    import uuid

    user_b = User(
        email=f"b-{uuid.uuid4().hex[:8]}@example.com",
        clerk_user_id=f"user_b_{uuid.uuid4().hex[:8]}",
    )
    session.add(user_b)
    await session.commit()
    await session.refresh(user_b)
    jwt_b = f"fake-jwt-for-{user_b.clerk_user_id}"

    r = await client.get(
        "/v1/keys", headers={"Authorization": f"Bearer {jwt_b}"}
    )
    assert r.status_code == 200
    assert r.json()["items"] == []

    # And user A still sees their own
    r2 = await client.get(
        "/v1/keys", headers={"Authorization": f"Bearer {fake_jwt}"}
    )
    assert len(r2.json()["items"]) == 1
