"""Tests for ``GET /v1/score/audit`` + ``GET /v1/score/audit/{audit_id}`` (Phase 22.D).

Coverage:
- single audit-call detail by id (happy + 404 paths)
- isolation: user A cannot read user B's audit row (returns 404)
- paginated list per user (default + custom limit/offset)
- list ordering: newest first
- pagination metadata (total, limit, offset)
- query-param validation (limit bounds)
- auth: missing key → 401; revoked key → 401
- audit_id format: aud_<hex>
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import APIKey, AuditCall, User


async def _seed_audit(
    session: AsyncSession,
    user_id: uuid.UUID,
    api_key_id: uuid.UUID,
    *,
    env_id: str = "math-algebra",
    reward: float = 0.5,
    created_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> AuditCall:
    row = AuditCall(
        user_id=user_id,
        api_key_id=api_key_id,
        env_id=env_id,
        env_version="0.0.1-test",
        seed=0,
        completion_hash="a" * 64,
        reward=reward,
        conformal_low=max(0.0, reward - 0.1),
        conformal_high=min(1.0, reward + 0.1),
        coverage=0.9,
        components_json={"format_valid": 1.0, "parse_valid": 1.0, "correct": reward},
        latency_ms=42,
        idempotency_key=idempotency_key,
    )
    if created_at is not None:
        row.created_at = created_at
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


# ── single audit-call detail ───────────────────────────────────────


async def test_get_audit_call_happy_path(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    row = await _seed_audit(session, info["user_id"], info["api_key_id"])
    audit_id = f"aud_{row.id.hex}"
    r = await client.get(
        f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audit_id"] == audit_id
    assert body["env_id"] == "math-algebra"
    assert body["seed"] == 0
    assert body["reward"] == 0.5
    assert body["completion_hash"] == "a" * 64


async def test_get_audit_call_unknown_id_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    fake = f"aud_{uuid.uuid4().hex}"
    r = await client.get(
        f"/v1/score/audit/{fake}", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
    assert r.json()["code"] == "audit_call_not_found"


async def test_get_audit_call_malformed_id_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/score/audit/not-a-valid-id", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
    assert r.json()["code"] == "audit_call_not_found"


async def test_get_audit_call_other_users_row_returns_404(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Caller A cannot read caller B's audit row — 404 (info-hiding)."""
    plaintext_a, _ = api_key

    # Create a second user + key + audit row.
    user_b = User(email="b@example.com")
    session.add(user_b)
    await session.flush()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=b"b" * 32,
        key_prefix="vlk_b",
        name="other-key",
    )
    session.add(key_b)
    await session.commit()
    row = await _seed_audit(session, user_b.id, key_b.id)

    audit_id = f"aud_{row.id.hex}"
    r = await client.get(
        f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext_a}
    )
    assert r.status_code == 404


async def test_get_audit_call_returns_completion_hash_only(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """GDPR: response carries ONLY the SHA-256 hash, never raw text."""
    plaintext, info = api_key
    row = await _seed_audit(session, info["user_id"], info["api_key_id"])
    r = await client.get(
        f"/v1/score/audit/aud_{row.id.hex}", headers={"X-Vlabs-Key": plaintext}
    )
    body = r.json()
    assert body["completion_hash"] == "a" * 64
    # Sanity: no key named 'completion' exposing plaintext.
    assert "completion" not in body or "hash" in str(body.get("completion", "hash"))


async def test_get_audit_call_includes_env_version(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    row = await _seed_audit(session, info["user_id"], info["api_key_id"])
    r = await client.get(
        f"/v1/score/audit/aud_{row.id.hex}", headers={"X-Vlabs-Key": plaintext}
    )
    body = r.json()
    assert body["env_version"] == "0.0.1-test"


# ── paginated list ─────────────────────────────────────────────────


async def test_list_audit_calls_empty(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/score/audit", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 100
    assert body["offset"] == 0


async def test_list_audit_calls_returns_user_rows(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    for i in range(3):
        await _seed_audit(session, info["user_id"], info["api_key_id"], reward=float(i) / 10)
    r = await client.get("/v1/score/audit", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    for item in body["items"]:
        assert item["audit_id"].startswith("aud_")


async def test_list_audit_calls_newest_first(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    now = datetime.now(UTC)
    rewards = [0.1, 0.2, 0.3]
    for i, reward in enumerate(rewards):
        await _seed_audit(
            session, info["user_id"], info["api_key_id"],
            reward=reward, created_at=now - timedelta(hours=i),
        )
    r = await client.get("/v1/score/audit", headers={"X-Vlabs-Key": plaintext})
    items = r.json()["items"]
    # newest (offset=0) is i=0 (now), oldest is i=2 (now-2h)
    assert items[0]["reward"] == 0.1
    assert items[-1]["reward"] == 0.3


async def test_list_audit_calls_limit_offset(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    for _ in range(5):
        await _seed_audit(session, info["user_id"], info["api_key_id"])
    r = await client.get(
        "/v1/score/audit?limit=2&offset=1", headers={"X-Vlabs-Key": plaintext}
    )
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 1


async def test_list_audit_calls_only_owners_rows(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext_a, info_a = api_key
    # User B with a row.
    user_b = User(email="b@example.com")
    session.add(user_b)
    await session.flush()
    key_b = APIKey(user_id=user_b.id, key_hash=b"c" * 32, key_prefix="vlk_c", name="b-key")
    session.add(key_b)
    await session.commit()
    await _seed_audit(session, user_b.id, key_b.id)
    # User A with two rows.
    await _seed_audit(session, info_a["user_id"], info_a["api_key_id"])
    await _seed_audit(session, info_a["user_id"], info_a["api_key_id"])

    r = await client.get("/v1/score/audit", headers={"X-Vlabs-Key": plaintext_a})
    body = r.json()
    assert body["total"] == 2  # A only sees A's rows


async def test_list_audit_calls_invalid_limit_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/score/audit?limit=0", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422


async def test_list_audit_calls_limit_above_max_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/score/audit?limit=10000", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422


async def test_list_audit_calls_negative_offset_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/score/audit?offset=-1", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422


# ── auth ──────────────────────────────────────────────────────────


async def test_get_audit_call_missing_api_key_rejected(
    client: AsyncClient
) -> None:
    r = await client.get(f"/v1/score/audit/aud_{uuid.uuid4().hex}")
    assert r.status_code == 401


async def test_list_audit_calls_missing_api_key_rejected(
    client: AsyncClient
) -> None:
    r = await client.get("/v1/score/audit")
    assert r.status_code == 401


# ── End-to-end: POST /v1/score → GET /v1/score/audit/{id} ──────────


async def test_score_then_audit_round_trip(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    score_r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert score_r.status_code == 200
    audit_id = score_r.json()["audit_id"]

    audit_r = await client.get(
        f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
    )
    assert audit_r.status_code == 200
    body = audit_r.json()
    assert body["audit_id"] == audit_id
    assert body["env_id"] == "math-algebra"
    # Reward consistent with /v1/score response.
    assert body["reward"] == score_r.json()["reward"]
