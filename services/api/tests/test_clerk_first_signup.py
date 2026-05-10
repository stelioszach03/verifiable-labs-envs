"""Tests for ``vlabs_api.clerk_auth._resolve_user`` first-signup race.

Phase 31.F post-deploy validation surfaced /v1/keys returning 500 on
the first dashboard hit by a brand-new Clerk user. Root cause: two
concurrent management-plane requests for the same fresh ``clerk_user_id``
both miss the initial SELECT and race to INSERT — the loser hits a
UNIQUE-constraint violation on ``clerk_user_id`` and bubbles a 500.
The same shape applies when two Clerk identities share an email.

The fix is the standard look-insert-look-again pattern: catch
IntegrityError on commit, roll back, re-SELECT by clerk_user_id OR
email, and return whoever won the race.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from vlabs_api import clerk_auth
from vlabs_api.db import APIKey, User

# ── helpers ────────────────────────────────────────────────────────


def _new_clerk_id() -> str:
    return f"user_{uuid.uuid4().hex[:16]}"


def _stub_jwt_for(monkeypatch, claims: dict[str, Any]) -> str:
    """Patch ``_verify_jwt`` to return ``claims`` on any token + return a
    sentinel bearer header value the route can include."""

    def _fake_verify(token: str) -> dict[str, Any]:
        return claims

    monkeypatch.setattr(clerk_auth, "_verify_jwt", _fake_verify)
    return "Bearer fake-jwt"


# ── happy-path first-signup ────────────────────────────────────────


async def test_first_signup_get_keys_creates_user_row_and_returns_empty(
    client, session, monkeypatch
) -> None:
    """A GET /v1/keys with a fresh Clerk JWT auto-creates the User row
    and returns an empty list (no keys yet)."""
    clerk_id = _new_clerk_id()
    bearer = _stub_jwt_for(
        monkeypatch,
        {"sub": clerk_id, "email": f"{clerk_id}@x.test", "exp": 9999999999, "iat": 0},
    )

    # Sanity: row doesn't exist yet.
    pre = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    assert pre.scalar_one_or_none() is None

    r = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r.status_code == 200, r.text
    assert r.json()["items"] == []

    post = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    user = post.scalar_one_or_none()
    assert user is not None
    assert user.email == f"{clerk_id}@x.test"


async def test_first_signup_post_keys_creates_user_and_returns_key(
    client, session, monkeypatch
) -> None:
    """A POST /v1/keys on a brand-new Clerk JWT mints a key AND creates
    the User row in the same request — no separate provisioning step."""
    clerk_id = _new_clerk_id()
    bearer = _stub_jwt_for(
        monkeypatch,
        {"sub": clerk_id, "email": f"{clerk_id}@x.test", "exp": 9999999999, "iat": 0},
    )

    r = await client.post(
        "/v1/keys", json={"name": "first-key"}, headers={"Authorization": bearer}
    )
    assert r.status_code == 200, r.text
    assert r.json()["plaintext_key"].startswith("vlk_")

    keys = await session.execute(select(APIKey))
    rows = keys.scalars().all()
    assert len(rows) == 1
    users = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    assert users.scalar_one_or_none() is not None


# ── race protection ────────────────────────────────────────────────


async def test_concurrent_first_signup_does_not_500_on_clerk_id_clash(
    client, session, monkeypatch
) -> None:
    """Simulate the race: another worker has already INSERTed the same
    clerk_user_id between our SELECT and our COMMIT. The fix re-SELECTs
    on IntegrityError and returns the winner row instead of bubbling 500.
    """
    clerk_id = _new_clerk_id()
    bearer = _stub_jwt_for(
        monkeypatch,
        {"sub": clerk_id, "email": f"{clerk_id}@x.test", "exp": 9999999999, "iat": 0},
    )

    # Pre-create the winner row directly — simulates the parallel worker.
    winner = User(
        email=f"{clerk_id}@x.test",
        name="winner",
        clerk_user_id=clerk_id,
    )
    session.add(winner)
    await session.commit()
    await session.refresh(winner)

    r = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r.status_code == 200, r.text
    assert r.json()["items"] == []

    # Only one row in the table — the IntegrityError path didn't
    # silently INSERT a duplicate.
    rows = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    assert len(rows.scalars().all()) == 1


async def test_first_signup_uses_jwt_email_when_present(
    client, session, monkeypatch
) -> None:
    """The User's email comes from the JWT ``email`` claim, not the
    placeholder pattern, when the JWT supplies one."""
    clerk_id = _new_clerk_id()
    real_email = f"jdoe-{uuid.uuid4().hex[:6]}@example.com"
    bearer = _stub_jwt_for(
        monkeypatch,
        {"sub": clerk_id, "email": real_email, "exp": 9999999999, "iat": 0},
    )

    r = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r.status_code == 200, r.text

    res = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    user = res.scalar_one_or_none()
    assert user is not None
    assert user.email == real_email
    assert "@clerk.placeholder" not in user.email


async def test_first_signup_uses_placeholder_email_when_jwt_missing(
    client, session, monkeypatch
) -> None:
    """If the Clerk JWT template doesn't include ``email``, fall back
    to the ``{clerk_id}@clerk.placeholder`` synthetic so the User row
    can still be created (a webhook backfills it later)."""
    clerk_id = _new_clerk_id()
    bearer = _stub_jwt_for(
        monkeypatch, {"sub": clerk_id, "exp": 9999999999, "iat": 0}
    )

    r = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r.status_code == 200, r.text

    res = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    user = res.scalar_one_or_none()
    assert user is not None
    assert user.email == f"{clerk_id}@clerk.placeholder"


async def test_re_auth_same_clerk_id_does_not_create_duplicate(
    client, session, monkeypatch
) -> None:
    """Hitting /v1/keys twice with the same JWT must hit the same User
    row — no duplicate INSERT."""
    clerk_id = _new_clerk_id()
    bearer = _stub_jwt_for(
        monkeypatch,
        {"sub": clerk_id, "email": f"{clerk_id}@x.test", "exp": 9999999999, "iat": 0},
    )

    r1 = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r1.status_code == 200
    r2 = await client.get("/v1/keys", headers={"Authorization": bearer})
    assert r2.status_code == 200

    rows = await session.execute(
        select(User).where(User.clerk_user_id == clerk_id)
    )
    assert len(rows.scalars().all()) == 1
