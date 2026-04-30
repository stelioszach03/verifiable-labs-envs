"""X-Vlabs-Key middleware — header parsing + DB lookup."""
from __future__ import annotations

from datetime import UTC, datetime


async def test_missing_header(client) -> None:
    r = await client.get("/v1/usage")
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "invalid_api_key"
    assert body["type"].endswith("/invalid_api_key")
    assert r.headers["content-type"].startswith("application/problem+json")


async def test_malformed_prefix(client) -> None:
    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": "not_a_vlabs_key"})
    assert r.status_code == 401
    assert r.json()["code"] == "invalid_api_key"


async def test_unknown_key(client) -> None:
    r = await client.get(
        "/v1/usage", headers={"X-Vlabs-Key": "vlk_unknownkeyabcdefghijklmnop"}
    )
    assert r.status_code == 401


async def test_valid_key_returns_200(client, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200


async def test_revoked_key_rejected(client, api_key, session) -> None:
    from sqlalchemy import select

    from vlabs_api.db import APIKey

    plaintext, info = api_key
    res = await session.execute(select(APIKey).where(APIKey.id == info["api_key_id"]))
    row = res.scalar_one()
    row.revoked_at = datetime.now(UTC)
    await session.commit()

    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 401
