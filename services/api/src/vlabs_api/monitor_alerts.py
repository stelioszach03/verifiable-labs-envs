"""Alert dispatch for monitor runs (Phase 28.D).

PHASE_28_PLAN.md §11 (D7-A primary email + D7-D dashboard fallback +
D7-B Slack opt-in).

LOCAL_FAKE_EMAIL mode (env var set, OR ``VLABS_EMAIL_API_KEY`` empty)
writes ``.eml`` files under ``/tmp/vlabs-emails/<ts>.eml`` so tests
can assert delivery without a real provider; production deploy sets
the Resend / SES key via Fly secrets.

Alert dispatch is best-effort: a failed delivery is logged + persisted
in ``monitor_alerts.delivery_error`` but does NOT block the run-row
commit. Verdict ``ok`` runs do NOT trigger alerts (avoids
notification fatigue).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import get_settings
from vlabs_api.db import Monitor, MonitorAlert, MonitorRun

log = structlog.get_logger(__name__)

LOCAL_FAKE_EMAIL_DIR = Path("/tmp/vlabs-emails")


# ── helpers ─────────────────────────────────────────────────────────


def _is_fake_email_mode() -> bool:
    """Tests run in fake mode; production deploys set the API key."""
    settings = get_settings()
    if os.environ.get("VLABS_LOCAL_FAKE_EMAIL", "").lower() in ("1", "true"):
        return True
    return not bool(getattr(settings, "vlabs_email_api_key", "") or "")


def _email_from_address() -> str:
    return os.environ.get("VLABS_EMAIL_FROM", "alerts@vlabs.local")


def _format_summary_table(summary: dict[str, Any]) -> str:
    per_env = summary.get("per_env") or {}
    if not per_env:
        return "  (no episodes recorded)"
    lines = []
    for env_id in sorted(per_env):
        stats = per_env[env_id]
        lines.append(
            f"  {env_id}: n={stats.get('n', 0)} "
            f"mean={float(stats.get('mean_reward', 0.0)):.4f} "
            f"coverage={float(stats.get('coverage', 0.0)):.3f}"
        )
    return "\n".join(lines)


def _format_email_subject(monitor_name: str, verdict: str) -> str:
    tag = {
        "regressed": "[REGRESSED]",
        "warning": "[WARNING]",
        "ok": "[OK]",
    }.get(verdict, "[NOTICE]")
    return f"[vlabs] {tag} {monitor_name}"


def _format_email_body(
    *,
    monitor_name: str,
    monitor_id: str,
    run_id: str,
    verdict_payload: dict[str, Any],
    summary: dict[str, Any],
    dashboard_url: str | None = None,
    pdf_url: str | None = None,
) -> str:
    verdict = verdict_payload.get("verdict", "unknown")
    conformal = verdict_payload.get("conformal") or {}
    bootstrap = verdict_payload.get("bootstrap") or {}

    lines = [
        f"Monitor: {monitor_name} ({monitor_id})",
        f"Run: {run_id}",
        f"Verdict: {verdict.upper()}",
        "",
        "Conformal coverage:",
        f"  current  = {conformal.get('current')}",
        f"  baseline = {conformal.get('baseline')}",
        f"  delta_to_target = {conformal.get('delta_to_target')}",
        "",
        "Bootstrap reward delta:",
        f"  mean_delta = {bootstrap.get('mean_delta')}",
        f"  95% CI    = ({bootstrap.get('ci_low')}, {bootstrap.get('ci_high')})",
        f"  p_value   = {bootstrap.get('p_value')}",
        "",
        "Per-environment summary:",
        _format_summary_table(summary),
        "",
    ]
    if dashboard_url:
        lines.append(f"Dashboard: {dashboard_url}")
    if pdf_url:
        lines.append(f"PDF report: {pdf_url}")
    lines.append("")
    lines.append("— vlabs continuous monitoring")
    return "\n".join(lines)


# ── email sender (Resend-style) ────────────────────────────────────


async def send_email_alert(
    *,
    to_address: str,
    subject: str,
    body: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Dispatch a single email. Returns ``{success, error}``.

    Wire format mirrors Resend's REST API
    (`POST /emails`). Falls back to writing a `.eml` file in
    LOCAL_FAKE_EMAIL mode for tests / local dev.
    """
    if _is_fake_email_mode():
        LOCAL_FAKE_EMAIL_DIR.mkdir(parents=True, exist_ok=True)
        path = LOCAL_FAKE_EMAIL_DIR / f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.eml"
        eml = (
            f"From: {_email_from_address()}\n"
            f"To: {to_address}\n"
            f"Subject: {subject}\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            f"{body}\n"
        )
        path.write_text(eml, encoding="utf-8")
        return {"success": True, "error": None, "path": str(path)}

    settings = get_settings()
    api_key = getattr(settings, "vlabs_email_api_key", "") or ""
    payload = {
        "from": _email_from_address(),
        "to": [to_address],
        "subject": subject,
        "text": body,
    }
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=10.0)
    try:
        try:
            resp = await http_client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"transport: {type(exc).__name__}"}
        if resp.status_code >= 400:
            return {
                "success": False,
                "error": f"http_{resp.status_code}: {resp.text[:160]}",
            }
        return {"success": True, "error": None}
    finally:
        if own_client and http_client is not None:
            await http_client.aclose()


# ── slack sender ──────────────────────────────────────────────────


async def send_slack_alert(
    *,
    webhook_url: str,
    monitor_name: str,
    verdict: str,
    summary: dict[str, Any],
    verdict_payload: dict[str, Any],
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST a Slack-block payload to the customer's webhook URL."""
    if _is_fake_email_mode():
        # In fake mode, write the Slack payload alongside emails so tests
        # can grep for the dispatched content.
        LOCAL_FAKE_EMAIL_DIR.mkdir(parents=True, exist_ok=True)
        path = LOCAL_FAKE_EMAIL_DIR / f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.slack"
        path.write_text(
            json.dumps({
                "webhook_url": webhook_url,
                "monitor_name": monitor_name,
                "verdict": verdict,
                "summary": summary,
                "verdict_payload": verdict_payload,
            }),
            encoding="utf-8",
        )
        return {"success": True, "error": None, "path": str(path)}

    text = (
        f"*[vlabs] {verdict.upper()}* — {monitor_name}\n"
        f"```{_format_summary_table(summary)}```"
    )
    payload = {"text": text}
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=5.0)
    try:
        try:
            resp = await http_client.post(webhook_url, json=payload)
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"transport: {type(exc).__name__}"}
        if resp.status_code >= 400:
            return {
                "success": False,
                "error": f"http_{resp.status_code}: {resp.text[:160]}",
            }
        return {"success": True, "error": None}
    finally:
        if own_client and http_client is not None:
            await http_client.aclose()


# ── orchestrator ──────────────────────────────────────────────────


async def dispatch_monitor_alerts(
    session: AsyncSession,
    *,
    monitor: Monitor,
    run: MonitorRun,
    summary: dict[str, Any],
    verdict_payload: dict[str, Any],
    dashboard_url: str | None = None,
    pdf_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[MonitorAlert]:
    """Send all configured alerts for ``run``.

    ``verdict='ok'`` runs are no-ops (no email burst on healthy
    monitors). Returns the persisted ``monitor_alerts`` rows so the
    caller can assert delivery state in tests.
    """
    verdict = (verdict_payload or {}).get("verdict", "ok")
    if verdict == "ok":
        return []
    if not monitor.alert_channels:
        return []

    monitor_id_str = f"mon_{monitor.id.hex}"
    run_id_str = f"mr_{run.id.hex}"

    rows: list[MonitorAlert] = []
    for channel in monitor.alert_channels:
        ch_type = channel.get("type")
        alert = MonitorAlert(
            monitor_run_id=run.id,
            channel=ch_type,
            dispatched_at=datetime.now(UTC),
        )
        session.add(alert)
        await session.flush()

        if ch_type == "email":
            address = channel.get("address")
            if not address:
                alert.delivery_error = "missing_address"
                rows.append(alert)
                continue
            subject = _format_email_subject(monitor.name, verdict)
            body = _format_email_body(
                monitor_name=monitor.name,
                monitor_id=monitor_id_str,
                run_id=run_id_str,
                verdict_payload=verdict_payload,
                summary=summary,
                dashboard_url=dashboard_url,
                pdf_url=pdf_url,
            )
            outcome = await send_email_alert(
                to_address=address,
                subject=subject,
                body=body,
                http_client=http_client,
            )
            if outcome["success"]:
                alert.delivered_at = datetime.now(UTC)
            else:
                alert.delivery_error = (outcome.get("error") or "")[:1000]
        elif ch_type == "slack":
            webhook_url = channel.get("webhook_url")
            if not webhook_url:
                alert.delivery_error = "missing_webhook_url"
                rows.append(alert)
                continue
            outcome = await send_slack_alert(
                webhook_url=webhook_url,
                monitor_name=monitor.name,
                verdict=verdict,
                summary=summary,
                verdict_payload=verdict_payload,
                http_client=http_client,
            )
            if outcome["success"]:
                alert.delivered_at = datetime.now(UTC)
            else:
                alert.delivery_error = (outcome.get("error") or "")[:1000]
        else:
            alert.delivery_error = f"unknown_channel: {ch_type}"
        rows.append(alert)

    await session.commit()
    return rows


async def list_alerts_for_run(
    session: AsyncSession, run_id: uuid.UUID,
) -> list[MonitorAlert]:
    res = await session.execute(
        select(MonitorAlert).where(MonitorAlert.monitor_run_id == run_id)
    )
    return list(res.scalars().all())


__all__ = [
    "LOCAL_FAKE_EMAIL_DIR",
    "send_email_alert",
    "send_slack_alert",
    "dispatch_monitor_alerts",
    "list_alerts_for_run",
]
