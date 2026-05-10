"""``/v1/admin/attestations/*`` — admin review-board endpoints (Phase 31.E).

Two endpoints, all Clerk-authed + admin-allowlisted (mirrors the
existing admin.py pattern):

- ``POST /v1/admin/attestations/{id}/decisions`` — record an audit
  decision (approve / reject / request_more / revoke). Wraps the
  31.B service-layer ``record_audit_decision`` + drives the cert
  issuance + status transitions.
- ``GET /v1/admin/attestations/{id}/audit-trail`` — list all audit
  decisions for an attestation, DESC by decided_at.

Endpoint design notes
- Body for the POST is :class:`AttestationAdminDecisionRequest`
  (auditor_kind / auditor_label / decision / audit_summary).
- The admin's Clerk user_id is recorded as the ``auditor_user_id`` on
  the audit row so the trail is non-repudiable.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.db import User, get_db
from vlabs_api.ids import (
    encode_attestation_audit_id,
    parse_attestation_id,
)
from vlabs_api.routes.admin import require_admin
from vlabs_api.schemas import (
    AttestationAdminDecisionRequest,
    AttestationAuditEntry,
)

router = APIRouter(tags=["admin-attestations"])


# ── POST /v1/admin/attestations/{id}/decisions ────────────────────


@router.post(
    "/admin/attestations/{attestation_id}/decisions",
    response_model=AttestationAuditEntry,
    status_code=201,
)
async def admin_record_decision(
    attestation_id: str,
    payload: AttestationAdminDecisionRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> AttestationAuditEntry:
    """Record one auditor decision (approve / reject / request_more
    / revoke) on an attestation. The admin's Clerk user_id is
    captured for non-repudiation."""
    uid = parse_attestation_id(attestation_id)
    audit = await svc.record_audit_decision(
        session,
        attestation_id=uid,
        auditor_kind=payload.auditor_kind,
        auditor_user_id=user.id,
        auditor_label=payload.auditor_label,
        audit_summary=payload.audit_summary,
        decision=payload.decision,
    )
    await session.commit()
    return AttestationAuditEntry(
        id=encode_attestation_audit_id(audit.id),
        auditor_kind=audit.auditor_kind,  # type: ignore[arg-type]
        auditor_label=audit.auditor_label,
        decision=audit.decision,  # type: ignore[arg-type]
        audit_summary=audit.audit_summary or {},
        decided_at=audit.decided_at,
    )


# ── GET /v1/admin/attestations/{id}/audit-trail ───────────────────


@router.get(
    "/admin/attestations/{attestation_id}/audit-trail",
    response_model=list[AttestationAuditEntry],
)
async def admin_audit_trail(
    attestation_id: str,
    user: User = Depends(require_admin),  # noqa: ARG001 — auth gating only
    session: AsyncSession = Depends(get_db),
) -> list[AttestationAuditEntry]:
    """List all audit decisions for an attestation, DESC by
    decided_at."""
    uid = parse_attestation_id(attestation_id)
    rows = await svc.list_audit_trail(session, attestation_id=uid)
    return [
        AttestationAuditEntry(
            id=encode_attestation_audit_id(r.id),
            auditor_kind=r.auditor_kind,  # type: ignore[arg-type]
            auditor_label=r.auditor_label,
            decision=r.decision,  # type: ignore[arg-type]
            audit_summary=r.audit_summary or {},
            decided_at=r.decided_at,
        )
        for r in rows
    ]


__all__ = ["router"]
