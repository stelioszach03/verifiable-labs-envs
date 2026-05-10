"""Phase 31.C — audit-decision service unit tests.

Direct service-layer coverage for :func:`vlabs_api.attestation_service
.record_audit_decision` and :func:`list_audit_trail`. These functions
are the kernel that 31.E will wire to the admin review-board route;
this file pins the state-machine semantics so 31.E doesn't break them.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.db import (
    APIKey,
    Attestation,
    AttestationAudit,
    User,
)
from vlabs_api.errors import (
    AttestationInvalidState,
    AttestationNotFound,
)

# ── helpers ────────────────────────────────────────────────────────


async def _make_user_key(session: AsyncSession) -> tuple[User, APIKey]:
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )

    user = User(
        email=f"u-{uuid.uuid4().hex[:8]}@example.com", name="auditor-target"
    )
    session.add(user)
    await session.flush()
    plaintext = generate_plaintext_key()
    key_row = APIKey(
        user_id=user.id,
        key_hash=hash_plaintext_key(plaintext),
        key_prefix=key_prefix(plaintext),
        name="audit-key",
    )
    session.add(key_row)
    await session.commit()
    return user, key_row


async def _make_attestation(
    session: AsyncSession,
    *,
    status: str = "submitted",
    tier: str = "bronze",
    cycle: str = "annual",
) -> Attestation:
    user, key_row = await _make_user_key(session)
    outcome = await svc.create_draft(
        session,
        user_id=user.id,
        api_key_id=key_row.id,
        organization="ACME",
        scope_type="model",
        scope_subject="x",
        tier=tier,
        cycle=cycle,
        standards_requested=["iso_42001"],
    )
    row = outcome.attestation
    row.status = status
    await session.flush()
    await session.commit()
    return row


# ── approve from submitted ─────────────────────────────────────────


async def test_approve_from_submitted_stamps_serial_and_expires_at(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={"score": 0.95},
        decision="approve",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    assert refreshed.status == "approved"
    assert refreshed.cert_serial is not None
    assert refreshed.cert_serial.startswith("stub-")
    assert refreshed.issued_at is not None
    assert refreshed.expires_at is not None
    assert audit.decision == "approve"
    assert audit.auditor_kind == "vlabs"


async def test_approve_from_under_review(session: AsyncSession) -> None:
    att = await _make_attestation(session, status="under_review")
    await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="third_party",
        auditor_user_id=None,
        auditor_label="external-firm",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    assert refreshed.status == "approved"


async def test_approve_continuous_uses_395_day_lifetime(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(
        session, status="submitted", tier="gold", cycle="continuous"
    )
    await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="third_party",
        auditor_user_id=None,
        auditor_label="external",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    delta = refreshed.expires_at - refreshed.issued_at
    assert abs(delta - timedelta(days=395)) < timedelta(seconds=10)


async def test_approve_annual_uses_365_day_lifetime(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(
        session, status="submitted", tier="bronze", cycle="annual"
    )
    await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="self",
        auditor_user_id=None,
        auditor_label="customer",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    delta = refreshed.expires_at - refreshed.issued_at
    assert abs(delta - timedelta(days=365)) < timedelta(seconds=10)


async def test_approve_from_draft_raises_invalid_state(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="draft")
    with pytest.raises(AttestationInvalidState):
        await svc.record_audit_decision(
            session,
            attestation_id=att.id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="approve",
        )


async def test_approve_from_approved_raises_invalid_state(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="approved")
    with pytest.raises(AttestationInvalidState):
        await svc.record_audit_decision(
            session,
            attestation_id=att.id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="approve",
        )


async def test_approve_from_revoked_raises_invalid_state(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="revoked")
    with pytest.raises(AttestationInvalidState):
        await svc.record_audit_decision(
            session,
            attestation_id=att.id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="approve",
        )


# ── reject from submitted → withdrawn ──────────────────────────────


async def test_reject_from_submitted_marks_withdrawn(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={"reason": "missing-evidence"},
        decision="reject",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    assert refreshed.status == "withdrawn"
    assert audit.decision == "reject"


async def test_reject_from_approved_raises_invalid_state(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="approved")
    with pytest.raises(AttestationInvalidState):
        await svc.record_audit_decision(
            session,
            attestation_id=att.id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="reject",
        )


# ── request_more from submitted → draft ────────────────────────────


async def test_request_more_from_submitted_returns_to_draft(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={"missing": ["additional_logs"]},
        decision="request_more",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    assert refreshed.status == "draft"
    assert audit.decision == "request_more"


async def test_request_more_from_approved_raises_invalid_state(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="approved")
    with pytest.raises(AttestationInvalidState):
        await svc.record_audit_decision(
            session,
            attestation_id=att.id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="request_more",
        )


# ── auditor variants ───────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["self", "vlabs", "third_party"])
async def test_record_decision_supports_all_auditor_kinds(
    session: AsyncSession, kind: str
) -> None:
    att = await _make_attestation(session, status="submitted")
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind=kind,
        auditor_user_id=None,
        auditor_label=f"{kind}-label",
        audit_summary={"k": "v"},
        decision="approve",
    )
    await session.commit()
    assert audit.auditor_kind == kind


# ── missing attestation ────────────────────────────────────────────


async def test_record_decision_missing_attestation_raises(
    session: AsyncSession,
) -> None:
    bogus_id = uuid.uuid4()
    with pytest.raises(AttestationNotFound):
        await svc.record_audit_decision(
            session,
            attestation_id=bogus_id,
            auditor_kind="vlabs",
            auditor_user_id=None,
            auditor_label="vlabs",
            audit_summary={},
            decision="approve",
        )


# ── summary JSONB persistence ──────────────────────────────────────


async def test_audit_summary_jsonb_round_trip(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    summary = {
        "score": 0.92,
        "frameworks_passed": ["iso_42001", "soc2"],
        "notes": "passed all checks",
        "nested": {"latency_ms": 1230, "samples": 1000},
    }
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary=summary,
        decision="approve",
    )
    await session.commit()
    refreshed_audit = (
        await session.execute(
            select(AttestationAudit).where(AttestationAudit.id == audit.id)
        )
    ).scalar_one()
    assert refreshed_audit.audit_summary == summary


async def test_audit_auditor_user_id_persisted(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    auditor_user = User(
        email=f"auditor-{uuid.uuid4().hex[:8]}@example.com",
        name="Auditor",
    )
    session.add(auditor_user)
    await session.flush()
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=auditor_user.id,
        auditor_label="vlabs-staff",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    assert audit.auditor_user_id == auditor_user.id


# ── audit trail listing ────────────────────────────────────────────


async def test_list_audit_trail_empty(session: AsyncSession) -> None:
    att = await _make_attestation(session, status="draft")
    rows = await svc.list_audit_trail(session, attestation_id=att.id)
    assert rows == []


async def test_list_audit_trail_descending_order(
    session: AsyncSession,
) -> None:
    att = await _make_attestation(session, status="submitted")
    a1 = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs-1",
        audit_summary={"k": 1},
        decision="request_more",
    )
    await session.commit()
    # Re-submit so we can record a second audit row.
    refreshed = (
        await session.execute(
            select(Attestation).where(Attestation.id == att.id)
        )
    ).scalar_one()
    refreshed.status = "submitted"
    await session.flush()
    await session.commit()
    a2 = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs-2",
        audit_summary={"k": 2},
        decision="approve",
    )
    await session.commit()

    rows = await svc.list_audit_trail(session, attestation_id=att.id)
    assert len(rows) == 2
    # DESC by decided_at: a2 (later) first, a1 (earlier) second.
    assert rows[0].id == a2.id
    assert rows[1].id == a1.id


async def test_list_audit_trail_isolates_by_attestation_id(
    session: AsyncSession,
) -> None:
    att1 = await _make_attestation(session, status="submitted")
    att2 = await _make_attestation(session, status="submitted")
    await svc.record_audit_decision(
        session,
        attestation_id=att1.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={"on": "att1"},
        decision="approve",
    )
    await svc.record_audit_decision(
        session,
        attestation_id=att2.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={"on": "att2"},
        decision="approve",
    )
    await session.commit()

    rows1 = await svc.list_audit_trail(session, attestation_id=att1.id)
    rows2 = await svc.list_audit_trail(session, attestation_id=att2.id)
    assert len(rows1) == 1
    assert len(rows2) == 1
    assert rows1[0].audit_summary == {"on": "att1"}
    assert rows2[0].audit_summary == {"on": "att2"}


# ── decided_at stamping ────────────────────────────────────────────


async def test_decided_at_is_recent_utc(session: AsyncSession) -> None:
    att = await _make_attestation(session, status="submitted")
    before = datetime.now(UTC) - timedelta(seconds=5)
    audit = await svc.record_audit_decision(
        session,
        attestation_id=att.id,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    refreshed = (
        await session.execute(
            select(AttestationAudit).where(AttestationAudit.id == audit.id)
        )
    ).scalar_one()
    after = datetime.now(UTC) + timedelta(seconds=5)
    assert before <= refreshed.decided_at <= after
