"""Phase 31.E — admin review-board endpoint tests.

Mirrors test_admin.py's auth pattern: stub_clerk_verify monkeypatches
clerk_auth._verify_jwt; VLABS_ADMIN_CLERK_IDS env var carries the
allowlist.
"""
from __future__ import annotations

import base64

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import Attestation, AttestationAudit
from vlabs_api.ids import (
    parse_attestation_id,
)


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


async def _create_submitted(
    client: AsyncClient, plaintext: str
) -> str:
    """Create a Bronze attestation, upload artifacts, submit it.
    Returns the encoded attestation id (att_...)."""
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "ACME",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": ["iso_42001"],
        },
    )
    assert r.status_code == 201
    aid = r.json()["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await client.post(
            f"/v1/attestations/{aid}/artifacts",
            headers=_hdr(plaintext),
            json={
                "kind": kind,
                "filename": f"{kind}.pdf",
                "content_b64": _b64(b"evidence"),
            },
        )
    await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    return aid


# ── auth gating (mirrors test_admin.py) ────────────────────────────


async def test_admin_decision_rejects_missing_clerk_token(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-test",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    assert r.status_code == 401


async def test_admin_decision_rejects_non_allowlist_clerk_user(
    client: AsyncClient, api_key, clerk_user, stub_clerk_verify
) -> None:
    stub_clerk_verify()
    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    fake_jwt, _ = clerk_user
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-test",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    assert r.status_code == 403


# ── happy path: approve via admin endpoint ─────────────────────────


async def test_admin_approves_submitted_attestation(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
    session: AsyncSession,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-internal",
            "decision": "approve",
            "audit_summary": {"score": 0.95},
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["decision"] == "approve"
    assert body["auditor_kind"] == "vlabs"
    assert body["audit_summary"]["score"] == 0.95

    # Attestation row reflects the approval.
    uid = parse_attestation_id(aid)
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == uid)
        )
    ).scalar_one()
    assert refreshed.status == "approved"
    assert refreshed.cert_serial is not None


# ── reject + request_more flows ────────────────────────────────────


async def test_admin_rejects_submitted_attestation(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
    session: AsyncSession,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-internal",
            "decision": "reject",
            "audit_summary": {"reason": "missing-evidence"},
        },
    )
    assert r.status_code == 201
    uid = parse_attestation_id(aid)
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == uid)
        )
    ).scalar_one()
    assert refreshed.status == "withdrawn"


async def test_admin_request_more_returns_to_draft(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
    session: AsyncSession,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-internal",
            "decision": "request_more",
            "audit_summary": {"missing": ["additional_logs"]},
        },
    )
    assert r.status_code == 201
    uid = parse_attestation_id(aid)
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == uid)
        )
    ).scalar_one()
    assert refreshed.status == "draft"


# ── invalid state guard ────────────────────────────────────────────


async def test_admin_approve_from_draft_returns_409(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    # Draft (not submitted).
    r0 = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "ACME",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": [],
        },
    )
    aid = r0.json()["id"]

    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    assert r.status_code == 409


# ── audit-trail listing ────────────────────────────────────────────


async def test_admin_audit_trail_lists_decisions(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
    session: AsyncSession,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    # Record an approval via the admin POST endpoint.
    r1 = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs-1",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    assert r1.status_code == 201

    r2 = await client.get(
        f"/v1/admin/attestations/{aid}/audit-trail",
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r2.status_code == 200
    rows = r2.json()
    assert len(rows) == 1
    assert rows[0]["decision"] == "approve"
    assert rows[0]["auditor_label"] == "vlabs-1"


async def test_admin_audit_trail_rejects_missing_token(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.get(
        f"/v1/admin/attestations/{aid}/audit-trail"
    )
    assert r.status_code == 401


# ── auditor_user_id non-repudiation ────────────────────────────────


async def test_admin_decision_records_auditor_user_id(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
    session: AsyncSession,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    uid = parse_attestation_id(aid)
    audits = (
        await session.execute(
            select(AttestationAudit).where(
                AttestationAudit.attestation_id == uid
            )
        )
    ).scalars().all()
    assert len(audits) == 1
    assert audits[0].auditor_user_id == user.id


# ── invalid input validation ───────────────────────────────────────


async def test_admin_decision_invalid_decision_returns_422(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "vlabs",
            "auditor_label": "vlabs",
            "decision": "not_a_real_decision",
            "audit_summary": {},
        },
    )
    assert r.status_code == 422


async def test_admin_decision_invalid_auditor_kind_returns_422(
    client: AsyncClient,
    api_key,
    clerk_user,
    stub_clerk_verify,
    monkeypatch,
) -> None:
    stub_clerk_verify()
    fake_jwt, user = clerk_user
    monkeypatch.setenv("VLABS_ADMIN_CLERK_IDS", user.clerk_user_id)
    from vlabs_api.config import get_settings

    get_settings.cache_clear()

    plaintext, _ = api_key
    aid = await _create_submitted(client, plaintext)
    r = await client.post(
        f"/v1/admin/attestations/{aid}/decisions",
        headers={"Authorization": f"Bearer {fake_jwt}"},
        json={
            "auditor_kind": "spy",
            "auditor_label": "vlabs",
            "decision": "approve",
            "audit_summary": {},
        },
    )
    assert r.status_code == 422
