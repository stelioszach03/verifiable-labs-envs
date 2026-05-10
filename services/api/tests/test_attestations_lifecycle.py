"""Phase 31.C — full attestation lifecycle integration tests.

Exercises the cross-endpoint flows that the Phase 31.B per-endpoint
tests don't cover end-to-end:

- draft → upload artifacts → submit → audit-approve → renewal → revoke;
- per-tier evidence requirements (Bronze 3, Silver 4, Gold 5+);
- cycle enforcement (gold => continuous, bronze/silver => annual);
- expires_at lifetime selection (365 d annual vs 395 d continuous);
- post-approval mutation locks (no further metadata edits, no further
  artifacts, withdraw blocked);
- idempotent renewal re-issue + audit-trail capture.
"""
from __future__ import annotations

import base64
import uuid
from datetime import timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.db import (
    Attestation,
)
from vlabs_api.ids import parse_attestation_id

# ── helpers ────────────────────────────────────────────────────────


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


async def _create_draft(
    client: AsyncClient,
    plaintext: str,
    *,
    tier: str = "bronze",
    cycle: str = "annual",
    organization: str = "ACME",
    scope_subject: str = "acme-llm-v1",
    standards: list[str] | None = None,
) -> dict:
    payload = {
        "organization": organization,
        "scope_type": "model",
        "scope_subject": scope_subject,
        "tier": tier,
        "cycle": cycle,
        "standards_requested": standards or ["iso_42001"],
    }
    r = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=payload
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _upload_kind(
    client: AsyncClient,
    plaintext: str,
    attestation_id: str,
    kind: str,
    body: bytes = b"evidence",
) -> dict:
    r = await client.post(
        f"/v1/attestations/{attestation_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": kind,
            "filename": f"{kind}.pdf",
            "content_b64": _b64(body),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _submit(
    client: AsyncClient, plaintext: str, attestation_id: str
) -> dict:
    r = await client.patch(
        f"/v1/attestations/{attestation_id}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _force_approve(
    session: AsyncSession,
    attestation_id: str,
) -> Attestation:
    """Force-approve via the audit-decision service path so tests can
    exercise post-approval flows without spinning up the (Phase 31.E)
    review queue."""
    uid = parse_attestation_id(attestation_id)
    # Move to submitted first if still draft.
    row = (
        await session.execute(
            __import__("sqlalchemy").select(Attestation).where(
                Attestation.id == uid
            )
        )
    ).scalar_one()
    if row.status == "draft":
        row.status = "submitted"
        await session.flush()

    await svc.record_audit_decision(
        session,
        attestation_id=uid,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs-test",
        audit_summary={"score": 1.0},
        decision="approve",
    )
    await session.commit()
    return (
        await session.execute(
            __import__("sqlalchemy").select(Attestation).where(
                Attestation.id == uid
            )
        )
    ).scalar_one()


# ── happy path: bronze full lifecycle ──────────────────────────────


async def test_bronze_full_lifecycle(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    assert att["status"] == "draft"

    # Upload all 3 Bronze required artifacts.
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)

    # Submit -> submitted.
    submitted = await _submit(client, plaintext, aid)
    assert submitted["status"] == "submitted"
    assert submitted["artifact_count"] == 3

    # Force approve via service (Phase 31.E will wire the admin route).
    row = await _force_approve(session, aid)
    assert row.status == "approved"
    assert row.cert_serial is not None
    assert row.cert_serial.startswith("stub-")
    assert row.issued_at is not None
    assert row.expires_at is not None

    # Lifetime delta = 365 days for annual.
    delta = row.expires_at - row.issued_at
    assert abs(delta - timedelta(days=365)) < timedelta(seconds=10)


async def test_silver_full_lifecycle(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="silver")
    aid = att["id"]
    for kind in (
        "training_doc",
        "audit_report",
        "legal_signoff",
        "monitor_record",
    ):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    row = await _force_approve(session, aid)
    assert row.status == "approved"


async def test_gold_full_lifecycle(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="gold", cycle="continuous")
    aid = att["id"]
    for kind in (
        "training_doc",
        "audit_report",
        "legal_signoff",
        "monitor_record",
        "change_mgmt",
        "rm_record",
    ):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    row = await _force_approve(session, aid)
    assert row.status == "approved"
    # Lifetime delta = 395 days for continuous.
    delta = row.expires_at - row.issued_at
    assert abs(delta - timedelta(days=395)) < timedelta(seconds=10)


# ── per-tier evidence requirement enforcement ──────────────────────


async def test_bronze_submit_blocked_with_only_2_artifacts(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    await _upload_kind(client, plaintext, aid, "training_doc")
    await _upload_kind(client, plaintext, aid, "audit_report")
    # Missing legal_signoff.
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r.status_code == 409


async def test_silver_submit_blocked_without_monitor_record(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="silver")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r.status_code == 409


async def test_gold_submit_requires_rm_or_prm(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="gold", cycle="continuous")
    aid = att["id"]
    # All Gold-required *except* rm_record / prm_record.
    for kind in (
        "training_doc",
        "audit_report",
        "legal_signoff",
        "monitor_record",
        "change_mgmt",
    ):
        await _upload_kind(client, plaintext, aid, kind)
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r.status_code == 409


async def test_gold_accepts_prm_record_in_place_of_rm(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="gold", cycle="continuous")
    aid = att["id"]
    for kind in (
        "training_doc",
        "audit_report",
        "legal_signoff",
        "monitor_record",
        "change_mgmt",
        "prm_record",  # PRM substitutes for RM.
    ):
        await _upload_kind(client, plaintext, aid, kind)
    submitted = await _submit(client, plaintext, aid)
    assert submitted["status"] == "submitted"


# ── cycle enforcement ──────────────────────────────────────────────


async def test_create_rejects_gold_with_annual_cycle(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "X",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "gold",
            "cycle": "annual",
            "standards_requested": [],
        },
    )
    assert r.status_code == 409


async def test_create_rejects_bronze_with_continuous_cycle(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "X",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "continuous",
            "standards_requested": [],
        },
    )
    assert r.status_code == 409


# ── post-approval lockdown ─────────────────────────────────────────


async def test_post_approval_artifact_upload_blocked(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r = await client.post(
        f"/v1/attestations/{aid}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "extra.pdf",
            "content_b64": _b64(b"more"),
        },
    )
    assert r.status_code == 409


async def test_post_approval_metadata_patch_blocked(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"organization": "Renamed Corp"},
    )
    assert r.status_code == 409


async def test_post_approval_withdraw_succeeds(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Withdraw is allowed from any non-terminal status by 31.B service
    contract — including approved (effectively voluntary cessation,
    distinct from explicit revoke). 31.E will tighten this to require
    revoke instead, but for now we pin the permissive behavior."""
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "withdraw"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "withdrawn"


# ── renewal flow ───────────────────────────────────────────────────


async def test_renewal_after_approve_creates_first_cycle(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """First renewal after the initial issuance is cycle_number=1.
    The initial issuance itself doesn't create a renewal row — cycle 1
    is the first audit-driven renewal cycle."""
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext),
        json={"idempotency_key": "key-A"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["cycle_number"] == 1
    assert body["completed_at"] is None


async def test_renewal_idempotent_within_24h(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r1 = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext),
        json={"idempotency_key": "key-B"},
    )
    r2 = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext),
        json={"idempotency_key": "key-B"},
    )
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


async def test_renewal_distinct_keys_create_distinct_rows(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r1 = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext),
        json={"idempotency_key": "key-X"},
    )
    r2 = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext),
        json={"idempotency_key": "key-Y"},
    )
    assert r1.json()["id"] != r2.json()["id"]
    # First renewal = cycle 1, second renewal = cycle 2.
    assert r1.json()["cycle_number"] == 1
    assert r2.json()["cycle_number"] == 2


# ── revoke flow ────────────────────────────────────────────────────


async def test_revoke_from_approved_records_audit(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    await _force_approve(session, aid)

    r = await client.request(
        "DELETE",
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "model decommissioned"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "revoked"
    assert body["revocation_reason"] == "model decommissioned"

    audits = await svc.list_audit_trail(
        session, attestation_id=parse_attestation_id(aid)
    )
    revoke_audits = [a for a in audits if a.decision == "revoke"]
    assert len(revoke_audits) == 1
    assert revoke_audits[0].auditor_kind == "self"


async def test_revoke_from_draft_allowed(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    r = await client.request(
        "DELETE",
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "abandoned"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "revoked"


# ── withdraw flow ──────────────────────────────────────────────────


async def test_withdraw_from_draft_succeeds(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "withdraw"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "withdrawn"


# ── metadata patches in draft only ─────────────────────────────────


async def test_patch_metadata_in_draft_succeeds(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"organization": "ACME 2.0", "scope_subject": "acme-llm-v2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["organization"] == "ACME 2.0"
    assert body["scope_subject"] == "acme-llm-v2"


async def test_patch_metadata_after_submit_blocked(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    await _submit(client, plaintext, aid)
    r = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"organization": "ACME 2.0"},
    )
    assert r.status_code == 409


# ── owner isolation ────────────────────────────────────────────────


async def test_renewal_blocks_other_owner(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Owner B cannot renew owner A's approved attestation."""
    plaintext_a, _ = api_key
    att_a = await _create_draft(client, plaintext_a, tier="bronze")
    aid = att_a["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext_a, aid, kind)
    await _submit(client, plaintext_a, aid)
    await _force_approve(session, aid)

    # Create a second user + key.
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    from vlabs_api.db import APIKey, User

    user_b = User(email=f"b-{uuid.uuid4().hex[:8]}@example.com", name="B")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="b-key",
    )
    session.add(key_b)
    await session.commit()

    r = await client.post(
        f"/v1/attestations/{aid}/renew",
        headers=_hdr(plaintext_b),
        json={"idempotency_key": "key-B"},
    )
    assert r.status_code == 404


async def test_revoke_blocks_other_owner(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Owner B cannot revoke owner A's draft attestation."""
    plaintext_a, _ = api_key
    att_a = await _create_draft(client, plaintext_a, tier="bronze")
    aid = att_a["id"]

    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    from vlabs_api.db import APIKey, User

    user_b = User(email=f"b2-{uuid.uuid4().hex[:8]}@example.com", name="B2")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="b2-key",
    )
    session.add(key_b)
    await session.commit()

    r = await client.request(
        "DELETE",
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext_b),
        json={"revocation_reason": "hostile"},
    )
    assert r.status_code == 404


# ── allowlist regression ───────────────────────────────────────────


async def test_create_uses_unique_public_id_under_collisions(
    client: AsyncClient, api_key
) -> None:
    """Multiple drafts must each get a distinct vl-XXXXXXXX public_id."""
    plaintext, _ = api_key
    seen: set[str] = set()
    for i in range(5):
        att = await _create_draft(
            client, plaintext, tier="bronze", scope_subject=f"acme-{i}"
        )
        assert att["public_id"] not in seen
        seen.add(att["public_id"])


async def test_submit_returns_submitted_status_and_artifact_count(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    att = await _create_draft(client, plaintext, tier="bronze")
    aid = att["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await _upload_kind(client, plaintext, aid, kind)
    submitted = await _submit(client, plaintext, aid)
    assert submitted["status"] == "submitted"
    assert submitted["artifact_count"] == 3
