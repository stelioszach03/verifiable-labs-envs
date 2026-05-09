"""Read + update tests for ``/v1/monitors`` (Phase 28.B).

Covers GET list, GET detail, PATCH (partial update + token rotate +
pause + rebaseline), DELETE (soft delete).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


async def _create(client, plaintext, *, name: str | None = None,
                  episodes: int = 10, env: str = "math-algebra"):
    payload = dict(_VALID_PAYLOAD)
    payload["name"] = name or _VALID_PAYLOAD["name"]
    payload["episodes_per_env"] = episodes
    payload["env_subset"] = [env]
    res = await client.post("/v1/monitors", json=payload, headers=_hdr(plaintext))
    assert res.status_code == 201, res.text
    return res.json()["monitor_id"]


async def _promote_to_pro(session, user_id) -> None:
    from vlabs_api.db import Subscription

    session.add(
        Subscription(
            user_id=user_id,
            stripe_subscription_id=f"sub_test_pro_{user_id.hex[:8]}",
            tier="pro",
            status="active",
            current_period_start=datetime.now(UTC),
            current_period_end=datetime.now(UTC) + timedelta(days=30),
        )
    )
    await session.commit()


# ── GET /v1/monitors (list) ────────────────────────────────────────


async def test_list_monitors_empty_for_new_user(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.get("/v1/monitors", headers=_hdr(plaintext))
    assert res.status_code == 200
    body = res.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 25
    assert body["offset"] == 0


async def test_list_monitors_returns_caller_only(client, api_key, session) -> None:
    plaintext, info = api_key
    await _promote_to_pro(session, info["user_id"])
    await _create(client, plaintext, name="m-a")
    await _create(client, plaintext, name="m-b")

    res = await client.get("/v1/monitors", headers=_hdr(plaintext))
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    names = {item["name"] for item in body["items"]}
    assert names == {"m-a", "m-b"}


async def test_list_monitors_filters_by_status(
    client, api_key, session
) -> None:
    plaintext, info = api_key
    await _promote_to_pro(session, info["user_id"])
    mid = await _create(client, plaintext, name="m-a")
    await _create(client, plaintext, name="m-b")

    # Pause one.
    await client.patch(
        f"/v1/monitors/{mid}",
        json={"status": "paused"},
        headers=_hdr(plaintext),
    )

    paused = await client.get(
        "/v1/monitors?status=paused", headers=_hdr(plaintext)
    )
    assert paused.json()["total"] == 1
    assert paused.json()["items"][0]["status"] == "paused"
    active = await client.get(
        "/v1/monitors?status=active", headers=_hdr(plaintext)
    )
    assert active.json()["total"] == 1


async def test_list_monitors_paginated(client, api_key, session) -> None:
    plaintext, info = api_key
    await _promote_to_pro(session, info["user_id"])
    await _create(client, plaintext, name="m-a")
    await _create(client, plaintext, name="m-b")
    await _create(client, plaintext, name="m-c")

    page1 = await client.get(
        "/v1/monitors?limit=2&offset=0", headers=_hdr(plaintext)
    )
    page2 = await client.get(
        "/v1/monitors?limit=2&offset=2", headers=_hdr(plaintext)
    )
    assert page1.json()["total"] == 3
    assert len(page1.json()["items"]) == 2
    assert len(page2.json()["items"]) == 1


# ── GET /v1/monitors/{id} (detail) ─────────────────────────────────


async def test_get_monitor_detail_returns_full_shape(client, api_key) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    res = await client.get(f"/v1/monitors/{mid}", headers=_hdr(plaintext))
    assert res.status_code == 200
    body = res.json()
    assert body["monitor_id"] == mid
    assert body["model_endpoint"] == "https://api.openai.com/v1"
    assert body["model_name"] == "gpt-4o-mini"
    assert body["projected_monthly_episodes"] == 30 * 1 * 10
    # Auth token must NEVER appear in the response.
    assert _VALID_PAYLOAD["auth_token"] not in res.text


async def test_get_monitor_returns_404_for_unknown_id(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.get(
        "/v1/monitors/mon_deadbeefdeadbeefdeadbeefdeadbeef",
        headers=_hdr(plaintext),
    )
    assert res.status_code == 404
    assert res.json()["code"] == "monitor_not_found"


async def test_get_monitor_returns_404_for_other_users_monitor(
    client, api_key, session
) -> None:
    """Cross-user isolation: caller B cannot see caller A's monitor."""
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    from vlabs_api.db import APIKey, User

    plaintext_a, _ = api_key
    mid = await _create(client, plaintext_a)

    # Create a separate user + key.
    other = User(email="other@example.com", name="Other")
    session.add(other)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    session.add(
        APIKey(
            user_id=other.id,
            key_hash=hash_plaintext_key(plaintext_b),
            key_prefix=key_prefix(plaintext_b),
            name="b-key",
        )
    )
    await session.commit()
    res = await client.get(f"/v1/monitors/{mid}", headers=_hdr(plaintext_b))
    assert res.status_code == 404


async def test_get_monitor_rejects_malformed_id(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.get(
        "/v1/monitors/not-a-real-id", headers=_hdr(plaintext)
    )
    assert res.status_code == 404
    assert res.json()["code"] == "monitor_not_found"


# ── PATCH /v1/monitors/{id} ────────────────────────────────────────


async def test_patch_monitor_updates_name(client, api_key) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    res = await client.patch(
        f"/v1/monitors/{mid}",
        json={"name": "renamed"},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 200
    assert res.json()["name"] == "renamed"


async def test_patch_monitor_pauses_and_resumes(client, api_key) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    paused = await client.patch(
        f"/v1/monitors/{mid}",
        json={"status": "paused"},
        headers=_hdr(plaintext),
    )
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    resumed = await client.patch(
        f"/v1/monitors/{mid}",
        json={"status": "active"},
        headers=_hdr(plaintext),
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "active"


async def test_patch_monitor_rotates_auth_token(client, api_key) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    before = await client.get(f"/v1/monitors/{mid}", headers=_hdr(plaintext))
    fp_before = before.json()["auth_token_fingerprint"]

    res = await client.patch(
        f"/v1/monitors/{mid}",
        json={"auth_token": "sk-rotated-NEWTOKEN-XXXXXXXX"},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 200
    assert res.json()["auth_token_fingerprint"] != fp_before
    # Plaintext rotation token must not appear in response.
    assert "NEWTOKEN" not in res.text


async def test_patch_monitor_rebaseline_clears_baseline_run_id(
    client, api_key
) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    res = await client.patch(
        f"/v1/monitors/{mid}",
        json={"rebaseline": True},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 200
    assert res.json()["baseline_run_id"] is None


async def test_patch_monitor_rejects_invalid_status_transition(
    client, api_key
) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    res = await client.patch(
        f"/v1/monitors/{mid}",
        json={"status": "failed"},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 409
    assert res.json()["code"] == "monitor_invalid_state"


async def test_patch_monitor_recomputes_next_run_at_on_cadence_change(
    client, api_key, session
) -> None:
    plaintext, info = api_key
    await _promote_to_pro(session, info["user_id"])
    mid = await _create(client, plaintext, name="m-orig", episodes=10)
    before = await client.get(f"/v1/monitors/{mid}", headers=_hdr(plaintext))
    next_before = before.json()["next_run_at"]

    res = await client.patch(
        f"/v1/monitors/{mid}",
        json={"cadence": "weekly"},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 200
    assert res.json()["cadence"] == "weekly"
    # next_run_at must shift later (weekly anchor > daily anchor).
    next_after = res.json()["next_run_at"]
    assert next_after > next_before


async def test_patch_monitor_404_for_unknown_id(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.patch(
        "/v1/monitors/mon_deadbeefdeadbeefdeadbeefdeadbeef",
        json={"name": "noop"},
        headers=_hdr(plaintext),
    )
    assert res.status_code == 404


# ── DELETE /v1/monitors/{id} ───────────────────────────────────────


async def test_delete_monitor_marks_failed_status(client, api_key) -> None:
    plaintext, _ = api_key
    mid = await _create(client, plaintext)
    res = await client.delete(f"/v1/monitors/{mid}", headers=_hdr(plaintext))
    assert res.status_code == 204
    detail = await client.get(f"/v1/monitors/{mid}", headers=_hdr(plaintext))
    assert detail.json()["status"] == "failed"


async def test_delete_monitor_404_for_unknown(client, api_key) -> None:
    plaintext, _ = api_key
    res = await client.delete(
        "/v1/monitors/mon_deadbeefdeadbeefdeadbeefdeadbeef",
        headers=_hdr(plaintext),
    )
    assert res.status_code == 404


# ── cadence helpers ────────────────────────────────────────────────


def test_compute_next_run_at_daily_advances_one_day() -> None:
    from datetime import UTC, datetime, timedelta

    from vlabs_api.monitor_cadence import compute_next_run_at

    anchor = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    nxt = compute_next_run_at("daily", anchor=anchor)
    assert nxt == anchor + timedelta(days=1)


def test_compute_next_run_at_weekly_advances_seven_days() -> None:
    from datetime import UTC, datetime, timedelta

    from vlabs_api.monitor_cadence import compute_next_run_at

    anchor = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    nxt = compute_next_run_at("weekly", anchor=anchor)
    assert nxt == anchor + timedelta(days=7)


def test_compute_next_run_at_monthly_advances_thirty_days() -> None:
    from datetime import UTC, datetime, timedelta

    from vlabs_api.monitor_cadence import compute_next_run_at

    anchor = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    nxt = compute_next_run_at("monthly", anchor=anchor)
    assert nxt == anchor + timedelta(days=30)


def test_compute_next_run_at_unknown_cadence_raises() -> None:
    import pytest

    from vlabs_api.monitor_cadence import compute_next_run_at

    with pytest.raises(ValueError, match="unknown cadence"):
        compute_next_run_at("hourly")


def test_projected_monthly_episodes() -> None:
    from vlabs_api.monitor_cadence import projected_monthly_episodes

    assert projected_monthly_episodes("daily", 1, 10) == 30 * 1 * 10
    assert projected_monthly_episodes("weekly", 3, 30) == 4 * 3 * 30
    assert projected_monthly_episodes("monthly", 5, 50) == 1 * 5 * 50


def test_runs_per_month() -> None:
    from vlabs_api.monitor_cadence import runs_per_month

    assert runs_per_month("daily") == 30
    assert runs_per_month("weekly") == 4
    assert runs_per_month("monthly") == 1


# ── ID prefix helpers ──────────────────────────────────────────────


def test_encode_monitor_id_uses_prefix() -> None:
    import uuid

    from vlabs_api.ids import MONITOR_PREFIX, encode_monitor_id

    uid = uuid.UUID("12345678-1234-1234-1234-123456789abc")
    encoded = encode_monitor_id(uid)
    assert encoded.startswith(MONITOR_PREFIX)
    # 32 hex chars after the prefix.
    assert encoded == f"{MONITOR_PREFIX}{uid.hex}"
    assert len(encoded) == len(MONITOR_PREFIX) + 32


def test_parse_monitor_id_round_trip() -> None:
    import uuid

    from vlabs_api.ids import encode_monitor_id, parse_monitor_id

    uid = uuid.uuid4()
    assert parse_monitor_id(encode_monitor_id(uid)) == uid
    assert parse_monitor_id(uid.hex) == uid


def test_parse_monitor_id_rejects_garbage() -> None:
    import pytest

    from vlabs_api.errors import MonitorNotFound
    from vlabs_api.ids import parse_monitor_id

    with pytest.raises(MonitorNotFound):
        parse_monitor_id("not-a-real-id")


def test_encode_monitor_run_id_uses_prefix() -> None:
    import uuid

    from vlabs_api.ids import MONITOR_RUN_PREFIX, encode_monitor_run_id

    uid = uuid.uuid4()
    assert encode_monitor_run_id(uid).startswith(MONITOR_RUN_PREFIX)
