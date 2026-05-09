"""Tests for the Phase 28.D alert dispatch (email + Slack)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.monitor_alerts import (
    LOCAL_FAKE_EMAIL_DIR,
    dispatch_monitor_alerts,
    list_alerts_for_run,
    send_email_alert,
    send_slack_alert,
)


@pytest.fixture(autouse=True)
def fake_email_mode(monkeypatch):
    """Force LOCAL_FAKE_EMAIL mode + clean the eml output dir per test."""
    monkeypatch.setenv("VLABS_LOCAL_FAKE_EMAIL", "true")
    if LOCAL_FAKE_EMAIL_DIR.exists():
        shutil.rmtree(LOCAL_FAKE_EMAIL_DIR)
    yield
    if LOCAL_FAKE_EMAIL_DIR.exists():
        shutil.rmtree(LOCAL_FAKE_EMAIL_DIR)


def _build_monitor(*, user_id, api_key_id, alert_channels) -> Monitor:
    from datetime import UTC, datetime, timedelta

    return Monitor(
        user_id=user_id,
        api_key_id=api_key_id,
        name="qwen-prod",
        model_endpoint="https://fake.test/v1",
        model_name="gpt-4o-mini",
        auth_token_encrypted=encrypt_llm_api_key("sk-fake"),
        auth_token_fingerprint="abcd1234",
        cadence="daily",
        env_subset=["math-algebra"],
        episodes_per_env=2,
        alert_channels=list(alert_channels),
        status="active",
        next_run_at=datetime.now(UTC) + timedelta(hours=1),
    )


def _build_run(monitor_id) -> MonitorRun:
    from datetime import UTC, datetime

    return MonitorRun(
        monitor_id=monitor_id,
        scheduled_at=datetime.now(UTC),
        status="success",
        finished_at=datetime.now(UTC),
        trigger="scheduled",
    )


_VERDICT_REGRESSED = {
    "verdict": "regressed",
    "conformal": {
        "current": 0.78, "baseline": 0.90, "delta_to_target": -0.12,
    },
    "bootstrap": {
        "mean_delta": -0.4, "ci_low": -0.5, "ci_high": -0.3, "p_value": 0.001,
        "regressed": True,
    },
    "per_env_breakdown": [],
}

_VERDICT_OK = {
    "verdict": "ok",
    "conformal": {"current": 0.90, "baseline": 0.90, "delta_to_target": 0.0},
    "bootstrap": {
        "mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
        "regressed": False,
    },
    "per_env_breakdown": [],
}


# ── send_email_alert ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_email_alert_writes_eml_in_fake_mode() -> None:
    out = await send_email_alert(
        to_address="ops@example.com",
        subject="[vlabs] [REGRESSED] qwen-prod",
        body="some body",
    )
    assert out["success"] is True
    assert "path" in out
    assert Path(out["path"]).exists()
    text = Path(out["path"]).read_text(encoding="utf-8")
    assert "ops@example.com" in text
    assert "qwen-prod" in text


# ── send_slack_alert ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_slack_alert_writes_payload_in_fake_mode() -> None:
    out = await send_slack_alert(
        webhook_url="https://hooks.slack.com/services/AAA/BBB/CCC",
        monitor_name="qwen-prod",
        verdict="regressed",
        summary={"per_env": {}},
        verdict_payload=_VERDICT_REGRESSED,
    )
    assert out["success"] is True
    files = list(LOCAL_FAKE_EMAIL_DIR.glob("*.slack"))
    assert len(files) == 1


# ── dispatch_monitor_alerts ────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_emits_email_for_regressed_verdict(
    api_key, session,
) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[
            {"type": "email", "address": "ops@example.com"},
        ],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    rows = await dispatch_monitor_alerts(
        session,
        monitor=monitor,
        run=run,
        summary={"per_env": {}, "n_total": 0},
        verdict_payload=_VERDICT_REGRESSED,
    )
    assert len(rows) == 1
    assert rows[0].channel == "email"
    assert rows[0].delivered_at is not None
    assert rows[0].delivery_error is None


@pytest.mark.asyncio
async def test_dispatch_skips_ok_verdict(api_key, session) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[
            {"type": "email", "address": "ops@example.com"},
        ],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    rows = await dispatch_monitor_alerts(
        session,
        monitor=monitor,
        run=run,
        summary={"per_env": {}, "n_total": 0},
        verdict_payload=_VERDICT_OK,
    )
    assert rows == []
    # Also: no .eml files written.
    if LOCAL_FAKE_EMAIL_DIR.exists():
        assert not list(LOCAL_FAKE_EMAIL_DIR.glob("*.eml"))


@pytest.mark.asyncio
async def test_dispatch_handles_multi_channel(api_key, session) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[
            {"type": "email", "address": "ops@example.com"},
            {"type": "slack", "webhook_url": "https://hooks.slack.com/AAA/BBB/CCC"},
        ],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    rows = await dispatch_monitor_alerts(
        session,
        monitor=monitor,
        run=run,
        summary={"per_env": {}, "n_total": 0},
        verdict_payload=_VERDICT_REGRESSED,
    )
    assert len(rows) == 2
    channels = {r.channel for r in rows}
    assert channels == {"email", "slack"}
    for r in rows:
        assert r.delivered_at is not None


@pytest.mark.asyncio
async def test_dispatch_records_missing_address_error(api_key, session) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[{"type": "email"}],  # no address
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    rows = await dispatch_monitor_alerts(
        session,
        monitor=monitor,
        run=run,
        summary={"per_env": {}, "n_total": 0},
        verdict_payload=_VERDICT_REGRESSED,
    )
    assert len(rows) == 1
    # Missing address path: no email sent, error recorded.
    assert rows[0].delivery_error == "missing_address"


@pytest.mark.asyncio
async def test_dispatch_no_channels_returns_empty(api_key, session) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    rows = await dispatch_monitor_alerts(
        session, monitor=monitor, run=run,
        summary={"per_env": {}}, verdict_payload=_VERDICT_REGRESSED,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_list_alerts_for_run_returns_persisted_rows(
    api_key, session,
) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[{"type": "email", "address": "ops@example.com"}],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    await dispatch_monitor_alerts(
        session, monitor=monitor, run=run,
        summary={"per_env": {}}, verdict_payload=_VERDICT_REGRESSED,
    )
    listed = await list_alerts_for_run(session, run.id)
    assert len(listed) == 1


# ── R10: alert payload must NOT carry the customer auth token ─────


@pytest.mark.asyncio
async def test_dispatched_email_does_not_leak_auth_token(
    api_key, session,
) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[{"type": "email", "address": "ops@example.com"}],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    session.add(run)
    await session.commit()

    await dispatch_monitor_alerts(
        session, monitor=monitor, run=run,
        summary={"per_env": {}}, verdict_payload=_VERDICT_REGRESSED,
    )
    files = list(LOCAL_FAKE_EMAIL_DIR.glob("*.eml"))
    assert files
    eml_text = files[0].read_text(encoding="utf-8")
    # Customer's plaintext key was "sk-fake" — must not appear.
    assert "sk-fake" not in eml_text
    # Encrypted Fernet bytes (or the FAKE marker) must not appear either.
    assert "FAKE::" not in eml_text


# ── worker integration: end-to-end run sets verdict + dispatches alert ─


@pytest.mark.asyncio
async def test_worker_run_persists_verdict_payload(
    api_key, session, monkeypatch,
) -> None:
    """End-to-end: worker run on a baseline-less monitor persists a
    regression_verdict (probably 'ok' or 'warning' against target=0.90)."""
    import httpx

    from vlabs_api import db as db_module
    from vlabs_api.monitor_worker import process_monitor_run

    transport_response_content = '{"answer": "0"}'

    class _Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                status_code=200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": transport_response_content,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

    real_async_client = httpx.AsyncClient

    def _fake_async_client(*args, **kwargs):
        kwargs["transport"] = _Transport()
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _fake_async_client)

    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        alert_channels=[],
    )
    session.add(monitor)
    await session.flush()
    run = _build_run(monitor.id)
    run.status = "queued"
    run.finished_at = None
    session.add(run)
    await session.commit()
    run_id = run.id
    session.expire_all()

    await process_monitor_run(
        run_id, session_factory=db_module._SessionFactory,
    )

    async with db_module._SessionFactory() as s:
        from sqlalchemy import select

        res = await s.execute(
            select(MonitorRun).where(MonitorRun.id == run_id)
        )
        final = res.scalar_one()
        assert final.status == "success"
        assert final.regression_verdict in ("ok", "warning", "regressed")
        assert final.verdict_payload is not None
        assert "conformal" in final.verdict_payload
        assert "bootstrap" in final.verdict_payload
