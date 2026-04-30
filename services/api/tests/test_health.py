"""GET /health — no auth, no DB hit."""
from __future__ import annotations


async def test_health_ok(client) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "dev"
    assert body["version"]


async def test_health_no_auth_required(client) -> None:
    """No X-Vlabs-Key header on /health — must still 200."""
    r = await client.get("/health")
    assert r.status_code == 200
