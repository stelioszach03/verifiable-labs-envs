"""Tests for the Phase 28.C monitor scheduler tick."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.monitor_scheduler import (
    schedule_manual_run,
    scheduler_tick,
)


def _build_monitor(*, user_id, api_key_id, next_run_at, status="active") -> Monitor:
    return Monitor(
        user_id=user_id,
        api_key_id=api_key_id,
        name=f"m-{user_id.hex[:6]}",
        model_endpoint="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        auth_token_encrypted=encrypt_llm_api_key("sk-fake-test"),
        auth_token_fingerprint="abcd1234",
        cadence="daily",
        env_subset=["math-algebra"],
        episodes_per_env=2,
        alert_channels=[],
        status=status,
        next_run_at=next_run_at,
    )


async def test_scheduler_tick_picks_due_monitors(session, api_key) -> None:
    """A monitor with next_run_at in the past must be enqueued."""
    _, info = api_key
    past = datetime.now(UTC) - timedelta(minutes=5)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=past,
    )
    session.add(monitor)
    await session.commit()

    enqueued = await scheduler_tick(session)
    assert len(enqueued) == 1

    res = await session.execute(MonitorRun.__table__.select())
    rows = res.fetchall()
    assert len(rows) == 1
    assert rows[0].status == "queued"
    assert rows[0].trigger == "scheduled"


async def test_scheduler_tick_skips_future_monitors(session, api_key) -> None:
    _, info = api_key
    future = datetime.now(UTC) + timedelta(hours=1)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=future,
    )
    session.add(monitor)
    await session.commit()

    enqueued = await scheduler_tick(session)
    assert enqueued == []


async def test_scheduler_tick_skips_paused_monitors(session, api_key) -> None:
    _, info = api_key
    past = datetime.now(UTC) - timedelta(minutes=5)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=past,
        status="paused",
    )
    session.add(monitor)
    await session.commit()

    enqueued = await scheduler_tick(session)
    assert enqueued == []


async def test_scheduler_tick_advances_next_run_at(session, api_key) -> None:
    """After firing, next_run_at must shift to a future timestamp."""
    _, info = api_key
    past = datetime.now(UTC) - timedelta(minutes=5)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=past,
    )
    session.add(monitor)
    await session.commit()

    await scheduler_tick(session)
    await session.refresh(monitor)
    assert monitor.next_run_at > datetime.now(UTC)
    assert monitor.last_run_at is not None


async def test_scheduler_tick_catch_up_fires_once(session, api_key) -> None:
    """A monitor 5 days late must fire ONCE on recovery, not 5 times."""
    _, info = api_key
    five_days_ago = datetime.now(UTC) - timedelta(days=5)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=five_days_ago,
    )
    session.add(monitor)
    await session.commit()

    enqueued = await scheduler_tick(session)
    assert len(enqueued) == 1
    # Tick again — must NOT enqueue a duplicate because next_run_at
    # was advanced past now.
    enqueued2 = await scheduler_tick(session)
    assert enqueued2 == []


async def test_scheduler_tick_idempotent_unique_constraint(
    session, api_key
) -> None:
    """Calling scheduler_tick twice without DB changes must not
    create duplicate runs at the same scheduled_at."""
    _, info = api_key
    past = datetime.now(UTC) - timedelta(minutes=5)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=past,
    )
    session.add(monitor)
    await session.commit()

    await scheduler_tick(session)
    # Force-set next_run_at back to the past — simulates a stale tick.
    monitor.next_run_at = past
    await session.commit()
    enqueued = await scheduler_tick(session)
    # The UNIQUE(monitor_id, scheduled_at) constraint blocks the dup.
    assert enqueued == []


async def test_schedule_manual_run_does_not_advance_next_run_at(
    session, api_key
) -> None:
    _, info = api_key
    future = datetime.now(UTC) + timedelta(hours=1)
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=future,
    )
    session.add(monitor)
    await session.commit()
    next_before = monitor.next_run_at

    run = await schedule_manual_run(session, monitor)
    assert run.trigger == "manual"
    assert run.status == "queued"
    await session.refresh(monitor)
    assert monitor.next_run_at == next_before


async def test_schedule_manual_run_collision_resolved(
    session, api_key
) -> None:
    """Two manual triggers in tight sequence both succeed."""
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"],
        api_key_id=info["api_key_id"],
        next_run_at=datetime.now(UTC) + timedelta(hours=1),
    )
    session.add(monitor)
    await session.commit()

    run_a = await schedule_manual_run(session, monitor)
    run_b = await schedule_manual_run(session, monitor)
    assert run_a.id != run_b.id


# ── HTTP-layer trigger ─────────────────────────────────────────────


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


_VALID_PAYLOAD = {
    "name": "qwen-prod",
    "model_endpoint": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
    "auth_token": "sk-test-customer-key-XXXXXXXXXXXXXXXX",
    "cadence": "daily",
    "env_subset": ["math-algebra"],
    "episodes_per_env": 5,
}


async def test_post_run_endpoint_creates_manual_run(client, api_key) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    assert create.status_code == 201, create.text
    monitor_id = create.json()["monitor_id"]

    res = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["status"] == "queued"
    assert body["trigger"] == "manual"
    assert body["monitor_run_id"].startswith("mr_")


async def test_post_run_endpoint_rejects_paused_monitor(
    client, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    await client.patch(
        f"/v1/monitors/{monitor_id}",
        json={"status": "paused"},
        headers=_hdr(plaintext),
    )
    res = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    assert res.status_code == 409
    assert res.json()["code"] == "monitor_invalid_state"


async def test_get_runs_list_after_manual_trigger(client, api_key) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    runs = await client.get(
        f"/v1/monitors/{monitor_id}/runs", headers=_hdr(plaintext)
    )
    assert runs.status_code == 200
    body = runs.json()
    assert body["total"] == 1
    assert body["items"][0]["trigger"] == "manual"
    assert body["items"][0]["status"] == "queued"


async def test_get_run_detail_404_for_unknown(client, api_key) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    res = await client.get(
        f"/v1/monitors/{monitor_id}/runs/mr_deadbeefdeadbeefdeadbeefdeadbeef",
        headers=_hdr(plaintext),
    )
    assert res.status_code == 404
    assert res.json()["code"] == "monitor_run_not_found"


async def test_get_run_detail_returns_canonical_shape(client, api_key) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_VALID_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    trigger = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    run_id = trigger.json()["monitor_run_id"]

    res = await client.get(
        f"/v1/monitors/{monitor_id}/runs/{run_id}",
        headers=_hdr(plaintext),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["monitor_run_id"] == run_id
    assert body["monitor_id"] == monitor_id
    assert body["status"] == "queued"
    assert body["pdf_url"] is None
