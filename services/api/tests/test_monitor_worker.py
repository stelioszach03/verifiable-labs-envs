"""Tests for the Phase 28.C monitor worker pipeline."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.monitor_worker import (
    process_monitor_run,
    rescue_queued_runs,
    reset_stale_running,
)


def _fake_chat_completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 16,
            "total_tokens": 48,
        },
    }


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, content: str = '{"answer": "0"}') -> None:
        self.content = content
        self.calls: list[str] = []

    async def handle_async_request(self, request):
        self.calls.append(str(request.url))
        return httpx.Response(
            status_code=200, json=_fake_chat_completion(self.content)
        )


def _build_monitor(
    *, user_id, api_key_id, status="active", env_subset=None,
    episodes_per_env=2, next_run_at=None,
) -> Monitor:
    return Monitor(
        user_id=user_id,
        api_key_id=api_key_id,
        name=f"m-{user_id.hex[:6]}",
        model_endpoint="https://fake-llm.test/v1",
        model_name="gpt-4o-mini",
        auth_token_encrypted=encrypt_llm_api_key("sk-fake-test"),
        auth_token_fingerprint="abcd1234",
        cadence="daily",
        env_subset=list(env_subset or ["math-algebra"]),
        episodes_per_env=int(episodes_per_env),
        alert_channels=[],
        status=status,
        next_run_at=(next_run_at or datetime.now(UTC) + timedelta(hours=1)),
    )


# ── reset_stale_running ────────────────────────────────────────────


async def test_reset_stale_running_resets_old_running(
    session, api_key
) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
    )
    session.add(monitor)
    await session.flush()
    stale = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=datetime.now(UTC) - timedelta(hours=3),
        status="running",
        started_at=datetime.now(UTC) - timedelta(hours=2),
        trigger="scheduled",
    )
    session.add(stale)
    await session.commit()

    n = await reset_stale_running(session)
    assert n == 1
    await session.refresh(stale)
    assert stale.status == "failed"
    assert stale.error == "scheduler_lost_run"


async def test_reset_stale_running_leaves_fresh_alone(session, api_key) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
    )
    session.add(monitor)
    await session.flush()
    fresh = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=datetime.now(UTC),
        status="running",
        started_at=datetime.now(UTC),
        trigger="scheduled",
    )
    session.add(fresh)
    await session.commit()

    n = await reset_stale_running(session)
    assert n == 0
    await session.refresh(fresh)
    assert fresh.status == "running"


# ── rescue_queued_runs ─────────────────────────────────────────────


async def test_rescue_queued_runs_returns_count(session, api_key) -> None:
    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
    )
    session.add(monitor)
    await session.flush()
    for i in range(3):
        session.add(
            MonitorRun(
                monitor_id=monitor.id,
                scheduled_at=datetime.now(UTC) + timedelta(seconds=i),
                status="queued",
                trigger="scheduled",
            )
        )
    await session.commit()

    count = await rescue_queued_runs(session)
    assert count == 3


# ── process_monitor_run end-to-end ─────────────────────────────────


@pytest.fixture
def patched_http(monkeypatch):
    """Patch the LLM client's default httpx.AsyncClient construction so
    process_monitor_run uses our fake transport when http_client=None."""
    transport = _FakeTransport(content='{"answer": "0"}')

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return transport


@pytest.mark.asyncio
async def test_process_monitor_run_completes_and_persists_summary(
    api_key, session, patched_http
) -> None:
    from vlabs_api import db as db_module

    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        episodes_per_env=2,
    )
    session.add(monitor)
    await session.flush()
    run = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=datetime.now(UTC),
        status="queued",
        trigger="manual",
    )
    session.add(run)
    await session.commit()
    run_id = run.id
    monitor_id = monitor.id
    session.expire_all()

    await process_monitor_run(
        run_id, session_factory=db_module._SessionFactory,
    )

    # Re-fetch to inspect.
    async with db_module._SessionFactory() as s:
        from sqlalchemy import select

        res = await s.execute(select(MonitorRun).where(MonitorRun.id == run_id))
        final = res.scalar_one()
        assert final.status == "success"
        assert final.summary_stats is not None
        assert final.pdf_storage_key is not None
        assert final.pdf_sha256
        # D6-A: first successful run becomes baseline.
        mres = await s.execute(select(Monitor).where(Monitor.id == monitor_id))
        monitor_now = mres.scalar_one()
        assert monitor_now.baseline_run_id == run_id
        assert monitor_now.last_run_at is not None


@pytest.mark.asyncio
async def test_process_monitor_run_does_not_overwrite_baseline(
    api_key, session, patched_http
) -> None:
    """A second successful run keeps the original baseline."""
    from vlabs_api import db as db_module

    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
        episodes_per_env=1,
    )
    session.add(monitor)
    await session.flush()

    runs: list[MonitorRun] = []
    for i in range(2):
        r = MonitorRun(
            monitor_id=monitor.id,
            scheduled_at=datetime.now(UTC) + timedelta(seconds=i),
            status="queued",
            trigger="manual",
        )
        session.add(r)
        runs.append(r)
    await session.commit()
    first_id = runs[0].id
    second_id = runs[1].id
    monitor_id = monitor.id
    session.expire_all()

    await process_monitor_run(first_id, session_factory=db_module._SessionFactory)
    await process_monitor_run(second_id, session_factory=db_module._SessionFactory)

    async with db_module._SessionFactory() as s:
        from sqlalchemy import select

        m_res = await s.execute(
            select(Monitor).where(Monitor.id == monitor_id)
        )
        m = m_res.scalar_one()
        assert m.baseline_run_id == first_id  # NOT second_id


@pytest.mark.asyncio
async def test_process_monitor_run_marks_failed_on_decrypt_error(
    session, api_key
) -> None:
    """Garbled auth_token_encrypted => failed run, never crashes."""
    from vlabs_api import db as db_module

    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
    )
    # Replace the encrypted token with garbage Fernet bytes that
    # the marker-check accepts (FAKE::garbage) but isn't a valid token.
    # Specifically: pass non-Fernet ciphertext that triggers an exception
    # only if VLABS_DATA_LLM_KEY_ENCRYPTION is set. In our test env it's
    # set to a known string, so non-Fernet bytes will fail.
    monitor.auth_token_encrypted = b"not-a-valid-fernet-token-and-no-marker"
    session.add(monitor)
    await session.flush()
    run = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=datetime.now(UTC),
        status="queued",
        trigger="manual",
    )
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
        assert final.status == "failed"
        assert final.error and "decrypt" in final.error


@pytest.mark.asyncio
async def test_process_monitor_run_no_op_for_already_succeeded(
    api_key, session,
) -> None:
    from vlabs_api import db as db_module

    _, info = api_key
    monitor = _build_monitor(
        user_id=info["user_id"], api_key_id=info["api_key_id"],
    )
    session.add(monitor)
    await session.flush()
    run = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=datetime.now(UTC),
        status="success",
        finished_at=datetime.now(UTC),
        trigger="manual",
    )
    session.add(run)
    await session.commit()
    run_id = run.id

    # No exception, no state change.
    await process_monitor_run(
        run_id, session_factory=db_module._SessionFactory,
    )
    await session.refresh(run)
    assert run.status == "success"


@pytest.mark.asyncio
async def test_process_monitor_run_unknown_id_returns_silently() -> None:
    """Worker.process_monitor_run for a UUID with no row must NOT crash —
    just log and return. (Race: another worker already picked it up.)"""
    import uuid

    from vlabs_api import db as db_module

    # No row exists for this UUID; expect a silent no-op return.
    await process_monitor_run(
        uuid.uuid4(), session_factory=db_module._SessionFactory,
    )


# ── monitor_cadence smoke (used by the scheduler tests already, but
# duplicating one assertion here so 28.C suite stands alone) ──────


def test_monitor_cadence_runs_per_month_known_values() -> None:
    from vlabs_api.monitor_cadence import RUNS_PER_MONTH

    assert RUNS_PER_MONTH["daily"] == 30
    assert RUNS_PER_MONTH["weekly"] == 4
    assert RUNS_PER_MONTH["monthly"] == 1
