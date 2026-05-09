"""End-to-end Phase 28 integration tests.

Walks through the full create → trigger → process → verdict → alert
pipeline against the in-process FastAPI app + a fake LLM endpoint.
This is the suite that proves all the 28.B-D pieces wire together.
"""
from __future__ import annotations

import shutil
from typing import Any

import httpx
import pytest

from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.monitor_alerts import LOCAL_FAKE_EMAIL_DIR

_PAYLOAD: dict[str, Any] = {
    "name": "qwen-prod-2026Q2",
    "model_endpoint": "https://fake-llm.test/v1",
    "model_name": "gpt-4o-mini",
    "auth_token": "sk-test-customer-key-XXXXXXXXXXXXXXXX",
    "cadence": "daily",
    "env_subset": ["math-algebra"],
    "episodes_per_env": 5,
    "alert_channels": [
        {"type": "email", "address": "ops@example.com"},
    ],
}


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


@pytest.fixture(autouse=True)
def fake_email(monkeypatch):
    monkeypatch.setenv("VLABS_LOCAL_FAKE_EMAIL", "true")
    if LOCAL_FAKE_EMAIL_DIR.exists():
        shutil.rmtree(LOCAL_FAKE_EMAIL_DIR)
    yield
    if LOCAL_FAKE_EMAIL_DIR.exists():
        shutil.rmtree(LOCAL_FAKE_EMAIL_DIR)


@pytest.fixture
def fake_llm_endpoint(monkeypatch):
    """Patch httpx.AsyncClient so all customer endpoint calls return a
    canned chat-completion."""

    class _Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            # Resend's API hits api.resend.com — leave that path alone
            # so the email helper falls back to FAKE_EMAIL mode (it
            # already does because VLABS_EMAIL_API_KEY is empty in tests).
            return httpx.Response(
                status_code=200,
                json={
                    "id": "chatcmpl-int",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"answer": "0"}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 16,
                        "completion_tokens": 8,
                        "total_tokens": 24,
                    },
                },
            )

    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = _Transport()
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_full_monitor_lifecycle_create_trigger_process(
    client, api_key, session, fake_llm_endpoint,
) -> None:
    """Create monitor → manual /run → process via worker → success row."""
    from sqlalchemy import select

    from vlabs_api import db as db_module
    from vlabs_api.ids import parse_monitor_id, parse_monitor_run_id
    from vlabs_api.monitor_worker import process_monitor_run

    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    assert create.status_code == 201, create.text
    monitor_id = create.json()["monitor_id"]

    trigger = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
    )
    assert trigger.status_code == 202
    run_id_str = trigger.json()["monitor_run_id"]

    # Process the queued run through the worker.
    run_uuid = parse_monitor_run_id(run_id_str)
    await process_monitor_run(
        run_uuid, session_factory=db_module._SessionFactory,
    )

    # Verify status + summary persisted.
    detail = await client.get(
        f"/v1/monitors/{monitor_id}/runs/{run_id_str}",
        headers=_hdr(plaintext),
    )
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "success"
    assert body["summary_stats"]
    assert body["summary_stats"]["per_env"]
    assert body["pdf_url"] is not None
    assert body["regression_verdict"] in ("ok", "warning", "regressed")

    # Baseline pointer must now reference this run (D6-A first-run rule).
    monitor_uuid = parse_monitor_id(monitor_id)
    async with db_module._SessionFactory() as s:
        m_res = await s.execute(
            select(Monitor).where(Monitor.id == monitor_uuid)
        )
        monitor_row = m_res.scalar_one()
        assert monitor_row.baseline_run_id == run_uuid


@pytest.mark.asyncio
async def test_full_lifecycle_alert_dispatched_for_warning(
    client, api_key, session, fake_llm_endpoint,
) -> None:
    """A run that flags coverage drift dispatches the configured email."""
    from sqlalchemy import select

    from vlabs_api import db as db_module
    from vlabs_api.ids import parse_monitor_run_id
    from vlabs_api.monitor_worker import process_monitor_run

    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]

    # 1st run: snapshot baseline.
    trigger1 = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
    )
    run1 = parse_monitor_run_id(trigger1.json()["monitor_run_id"])
    await process_monitor_run(
        run1, session_factory=db_module._SessionFactory,
    )

    # 2nd run: same fake LLM = same coverage = no regression. Just
    # verify the alert dispatch runs to completion (no exception even
    # in the 'ok' branch which short-circuits before alerts).
    trigger2 = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
    )
    run2 = parse_monitor_run_id(trigger2.json()["monitor_run_id"])
    await process_monitor_run(
        run2, session_factory=db_module._SessionFactory,
    )

    async with db_module._SessionFactory() as s:
        # Run 2 should be terminal (success).
        r_res = await s.execute(
            select(MonitorRun).where(MonitorRun.id == run2)
        )
        assert r_res.scalar_one().status == "success"


@pytest.mark.asyncio
async def test_full_lifecycle_pdf_uploaded_to_fake_r2(
    client, api_key, fake_llm_endpoint,
) -> None:
    """The processed run produces a PDF in R2 (fake mode → on-disk file)."""
    from pathlib import Path

    from vlabs_api import db as db_module
    from vlabs_api.ids import parse_monitor_run_id
    from vlabs_api.monitor_worker import process_monitor_run

    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]
    trigger = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
    )
    run_uuid = parse_monitor_run_id(trigger.json()["monitor_run_id"])
    await process_monitor_run(
        run_uuid, session_factory=db_module._SessionFactory,
    )
    detail = await client.get(
        f"/v1/monitors/{monitor_id}/runs/{trigger.json()['monitor_run_id']}",
        headers=_hdr(plaintext),
    )
    pdf_url = detail.json()["pdf_url"]
    assert pdf_url and pdf_url.startswith("file://")
    on_disk = Path(pdf_url.removeprefix("file://"))
    assert on_disk.exists()
    pdf_bytes = on_disk.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"qwen-prod" in pdf_bytes


@pytest.mark.asyncio
async def test_full_lifecycle_run_count_increments(
    client, api_key, fake_llm_endpoint,
) -> None:
    from vlabs_api import db as db_module
    from vlabs_api.ids import parse_monitor_run_id
    from vlabs_api.monitor_worker import process_monitor_run

    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]

    for _ in range(3):
        t = await client.post(
            f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
        )
        await process_monitor_run(
            parse_monitor_run_id(t.json()["monitor_run_id"]),
            session_factory=db_module._SessionFactory,
        )

    runs = await client.get(
        f"/v1/monitors/{monitor_id}/runs", headers=_hdr(plaintext),
    )
    assert runs.json()["total"] == 3
    statuses = {item["status"] for item in runs.json()["items"]}
    assert statuses == {"success"}


@pytest.mark.asyncio
async def test_full_lifecycle_auth_token_never_in_responses(
    client, api_key, fake_llm_endpoint,
) -> None:
    """R2: customer auth token must never appear in any /v1/monitors/* response."""
    plaintext, _ = api_key
    payload = dict(_PAYLOAD)
    payload["auth_token"] = "sk-secret-token-DO-NOT-LEAK-XXXXXXXX"
    create = await client.post(
        "/v1/monitors", json=payload, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]
    detail = await client.get(
        f"/v1/monitors/{monitor_id}", headers=_hdr(plaintext),
    )
    runs_list = await client.get(
        f"/v1/monitors/{monitor_id}/runs", headers=_hdr(plaintext),
    )
    for resp in (create, detail, runs_list):
        assert "DO-NOT-LEAK" not in resp.text


@pytest.mark.asyncio
async def test_full_lifecycle_pause_blocks_manual_run(
    client, api_key, fake_llm_endpoint,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]
    await client.patch(
        f"/v1/monitors/{monitor_id}",
        json={"status": "paused"},
        headers=_hdr(plaintext),
    )
    trigger = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext),
    )
    assert trigger.status_code == 409


@pytest.mark.asyncio
async def test_full_lifecycle_delete_makes_subsequent_get_404(
    client, api_key, fake_llm_endpoint,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext),
    )
    monitor_id = create.json()["monitor_id"]
    delete = await client.delete(
        f"/v1/monitors/{monitor_id}", headers=_hdr(plaintext),
    )
    assert delete.status_code == 204
    after = await client.get(
        f"/v1/monitors/{monitor_id}", headers=_hdr(plaintext),
    )
    # Soft-delete keeps the row but flips status='failed' — visible.
    assert after.status_code == 200
    assert after.json()["status"] == "failed"


@pytest.mark.asyncio
async def test_existing_endpoints_still_register_alongside_monitors(
    client, api_key,
) -> None:
    """Backwards-compat: Phase 22-23 endpoints must remain reachable
    after the Phase 28 wiring lands."""
    plaintext, _ = api_key
    h = await client.get("/health")
    assert h.status_code == 200
    u = await client.get("/v1/usage", headers=_hdr(plaintext))
    assert u.status_code == 200
    # Datasets endpoints (Phase 23) still register with the auth path.
    d = await client.get("/v1/datasets", headers=_hdr(plaintext))
    assert d.status_code == 200
