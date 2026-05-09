"""``POST /v1/monitors`` tests (Phase 28.B)."""
from __future__ import annotations

from typing import Any

_VALID_PAYLOAD: dict[str, Any] = {
    "name": "qwen-prod-2026Q2",
    "model_endpoint": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
    "auth_token": "sk-test-customer-key-XXXXXXXXXXXXXXXX",
    "cadence": "daily",
    "env_subset": ["math-algebra"],
    "episodes_per_env": 10,
    "alert_channels": [{"type": "email", "address": "ops@example.com"}],
}


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


async def test_create_monitor_returns_201_and_canonical_shape(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["monitor_id"].startswith("mon_")
    assert body["status"] == "active"
    assert body["cadence"] == "daily"
    assert body["projected_monthly_episodes"] == 30 * 1 * 10  # daily × 1 env × 10
    assert body["tier_limit_episodes"] > 0
    assert body["auth_token_fingerprint"]
    # The next_run_at must be in the future.
    from datetime import UTC, datetime
    next_run_at = datetime.fromisoformat(body["next_run_at"].replace("Z", "+00:00"))
    assert next_run_at > datetime.now(UTC)


async def test_create_monitor_persists_encrypted_token(
    client, api_key, session
) -> None:
    """Auth token round-trips through Fernet at rest; plaintext never
    appears in the row's BYTEA, and the fingerprint matches."""
    from sqlalchemy import select

    from vlabs_api.db import Monitor
    from vlabs_api.llm_key_crypto import decrypt_llm_api_key

    plaintext, _ = api_key
    res = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert res.status_code == 201
    rows = (await session.execute(select(Monitor))).scalars().all()
    assert len(rows) == 1
    monitor = rows[0]
    # Plaintext never lives in the BYTEA blob.
    assert _VALID_PAYLOAD["auth_token"].encode("utf-8") not in monitor.auth_token_encrypted
    # But Fernet decrypts back to the original.
    assert decrypt_llm_api_key(monitor.auth_token_encrypted) == _VALID_PAYLOAD["auth_token"]


async def test_create_monitor_response_does_not_leak_plaintext_token(
    client, api_key
) -> None:
    plaintext, _ = api_key
    res = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    body_text = res.text
    assert _VALID_PAYLOAD["auth_token"] not in body_text


async def test_create_monitor_rejects_unknown_env(client, api_key) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["env_subset"] = ["does-not-exist"]
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 404
    assert res.json()["code"] == "unknown_environment"


async def test_create_monitor_rejects_invalid_cadence(client, api_key) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["cadence"] = "hourly"
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 422


async def test_create_monitor_rejects_episodes_above_tier_cap(
    client, api_key
) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["episodes_per_env"] = 200  # free cap = 10
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 402
    assert res.json()["code"] == "monitor_tier_exceeded"


async def test_create_monitor_rejects_envs_above_tier_cap(client, api_key) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["env_subset"] = ["math-algebra", "code-humaneval"]  # free cap = 1
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 402


async def test_create_monitor_rejects_above_active_count_cap(
    client, api_key
) -> None:
    """Free tier has monitors_max=1 — a second create must be rejected."""
    plaintext, _ = api_key
    res1 = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert res1.status_code == 201
    payload2 = dict(_VALID_PAYLOAD)
    payload2["name"] = "second-monitor"
    res2 = await client.post(
        "/v1/monitors", json=payload2, headers=_hdr(plaintext)
    )
    assert res2.status_code == 402
    assert res2.json()["code"] == "monitor_tier_exceeded"


async def test_create_monitor_rejects_duplicate_name(
    client, api_key, session
) -> None:
    """Promote the user to pro tier (monitors_max=3) so the count cap
    doesn't shadow the name-conflict path."""
    from datetime import UTC, datetime, timedelta

    from vlabs_api.db import Subscription

    plaintext, info = api_key
    session.add(
        Subscription(
            user_id=info["user_id"],
            stripe_subscription_id="sub_test_pro_12345",
            tier="pro",
            status="active",
            current_period_start=datetime.now(UTC),
            current_period_end=datetime.now(UTC) + timedelta(days=30),
        )
    )
    await session.commit()

    res1 = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert res1.status_code == 201, res1.text
    res2 = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert res2.status_code == 409
    assert res2.json()["code"] == "monitor_name_conflict"


async def test_create_monitor_requires_auth(client) -> None:
    res = await client.post("/v1/monitors", json=_VALID_PAYLOAD)
    assert res.status_code == 401


async def test_create_monitor_rejects_email_channel_without_address(
    client, api_key
) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["alert_channels"] = [{"type": "email"}]
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 409
    assert res.json()["code"] == "monitor_invalid_state"


async def test_create_monitor_rejects_slack_channel_without_url(
    client, api_key
) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["alert_channels"] = [{"type": "slack"}]
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 409


async def test_create_monitor_slack_channel_url_fingerprint_in_response(
    client, api_key
) -> None:
    plaintext, _ = api_key
    payload = dict(_VALID_PAYLOAD)
    payload["alert_channels"] = [
        {"type": "email", "address": "ops@example.com"},
        {"type": "slack", "webhook_url": "https://hooks.slack.com/services/AAA/BBB/CCC"},
    ]
    res = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext)
    )
    assert res.status_code == 201
    monitor_id = res.json()["monitor_id"]

    detail = await client.get(
        f"/v1/monitors/{monitor_id}", headers=_hdr(plaintext)
    )
    body = detail.json()
    channels = body["alert_channels"]
    slack = next(c for c in channels if c["type"] == "slack")
    assert "webhook_url_fingerprint" in slack
    assert slack["webhook_url_fingerprint"]
    # Raw URL must NOT come back over the wire.
    assert "AAA/BBB/CCC" not in detail.text
    assert "hooks.slack.com" not in detail.text
