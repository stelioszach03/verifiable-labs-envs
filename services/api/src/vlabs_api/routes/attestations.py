"""``/v1/attestations`` — V-Certified attestation owner endpoints (Phase 31.B).

Seven owner endpoints per :doc:`PHASE_31_PLAN.md` §6 D6. Public
verification endpoints (`/registry`, `/verify/*`, `/badge/*`) ship
in 31.D as a separate router.

- ``POST /v1/attestations`` — create draft.
- ``GET /v1/attestations`` — paginated list.
- ``GET /v1/attestations/{id}`` — single-attestation detail.
- ``PATCH /v1/attestations/{id}`` — update / submit / withdraw.
- ``POST /v1/attestations/{id}/artifacts`` — upload supporting evidence.
- ``POST /v1/attestations/{id}/renew`` — initiate renewal cycle (idempotent).
- ``DELETE /v1/attestations/{id}`` — request revocation.

Auth: ``X-Vlabs-Key`` (data-plane, mirrors Phases 22 + 29 + 30).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.db import (
    Attestation,
    AttestationArtifact,
    AttestationRenewal,
    get_db,
)
from vlabs_api.ids import (
    encode_attestation_artifact_id,
    encode_attestation_id,
    encode_attestation_renewal_id,
    parse_attestation_id,
)
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import (
    AttestationArtifactInfo,
    AttestationArtifactRequest,
    AttestationCreateRequest,
    AttestationInfo,
    AttestationList,
    AttestationPatchRequest,
    AttestationRenewalInfo,
    AttestationRenewalRequest,
    AttestationRevokeRequest,
    AttestationStandardsAlignment,
    AttestationStatus,
    AttestationSummary,
    AttestationTier,
)

router = APIRouter(tags=["attestations"])

DEFAULT_LIMIT: int = 25
MAX_LIMIT: int = 200


# ── helpers ─────────────────────────────────────────────────────────


def _attestation_to_info(
    row: Attestation, *, artifact_count: int
) -> AttestationInfo:
    alignment_payload = row.standards_alignment or {}
    return AttestationInfo(
        id=encode_attestation_id(row.id),
        public_id=row.public_id,
        organization=row.organization,
        scope_type=row.scope_type,  # type: ignore[arg-type]
        scope_subject=row.scope_subject,
        tier=row.tier,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        cycle=row.cycle,  # type: ignore[arg-type]
        issued_at=row.issued_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        revocation_reason=row.revocation_reason,
        cert_serial=row.cert_serial,
        standards_alignment=AttestationStandardsAlignment(
            standards=alignment_payload.get("standards", []) or [],
            crosswalk_version=alignment_payload.get("crosswalk_version"),
            framework_versions=alignment_payload.get(
                "framework_versions", {}
            )
            or {},
        ),
        artifact_count=int(artifact_count),
        created_at=row.created_at,
    )


def _attestation_to_summary(row: Attestation) -> AttestationSummary:
    return AttestationSummary(
        id=encode_attestation_id(row.id),
        public_id=row.public_id,
        organization=row.organization,
        scope_type=row.scope_type,  # type: ignore[arg-type]
        scope_subject=row.scope_subject,
        tier=row.tier,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        cycle=row.cycle,  # type: ignore[arg-type]
        issued_at=row.issued_at,
        expires_at=row.expires_at,
        created_at=row.created_at,
    )


def _artifact_to_info(row: AttestationArtifact) -> AttestationArtifactInfo:
    return AttestationArtifactInfo(
        id=encode_attestation_artifact_id(row.id),
        attestation_id=encode_attestation_id(row.attestation_id),
        kind=row.kind,  # type: ignore[arg-type]
        storage_uri=row.storage_uri,
        sha256_hash=row.sha256_hash,
        encrypted=row.encrypted,
        size_bytes=int(row.size_bytes),
        submitted_at=row.submitted_at,
    )


def _renewal_to_info(row: AttestationRenewal) -> AttestationRenewalInfo:
    return AttestationRenewalInfo(
        id=encode_attestation_renewal_id(row.id),
        attestation_id=encode_attestation_id(row.attestation_id),
        cycle_number=int(row.cycle_number),
        initiated_at=row.initiated_at,
        completed_at=row.completed_at,
        new_cert_serial=row.new_cert_serial,
    )


# ── POST /v1/attestations ──────────────────────────────────────────


@router.post(
    "/attestations", response_model=AttestationInfo, status_code=201
)
async def create_attestation(
    payload: AttestationCreateRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AttestationInfo:
    """Create a draft attestation. Customer must subsequently upload
    required artifacts (per D9 + tier) and PATCH with action=submit
    to advance the lifecycle."""
    outcome = await svc.create_draft(
        session,
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        organization=payload.organization,
        scope_type=payload.scope_type,
        scope_subject=payload.scope_subject,
        tier=payload.tier,
        cycle=payload.cycle,
        standards_requested=payload.standards_requested,
    )
    await session.commit()
    return _attestation_to_info(outcome.attestation, artifact_count=0)


# ── GET /v1/attestations ───────────────────────────────────────────


@router.get("/attestations", response_model=AttestationList)
async def list_attestations(
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    status: AttestationStatus | None = Query(default=None),
    tier: AttestationTier | None = Query(default=None),
) -> AttestationList:
    """Paginated owner-facing list."""
    rows, total = await svc.list_for_owner(
        session,
        user_id=auth.user_id,
        limit=limit,
        offset=offset,
        status=status,
        tier=tier,
    )
    return AttestationList(
        items=[_attestation_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# ── GET /v1/attestations/{id} ──────────────────────────────────────


@router.get(
    "/attestations/{attestation_id}", response_model=AttestationInfo
)
async def get_attestation(
    attestation_id: str,
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_db),
) -> AttestationInfo:
    """Single-attestation detail (owner only)."""
    uid = parse_attestation_id(attestation_id)
    row = await svc.get_for_owner(
        session, attestation_id=uid, user_id=auth.user_id
    )
    artifact_count = await svc.count_artifacts(
        session, attestation_id=row.id
    )
    return _attestation_to_info(row, artifact_count=artifact_count)


# ── PATCH /v1/attestations/{id} ────────────────────────────────────


@router.patch(
    "/attestations/{attestation_id}", response_model=AttestationInfo
)
async def patch_attestation(
    attestation_id: str,
    payload: AttestationPatchRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AttestationInfo:
    """Update / submit / withdraw."""
    uid = parse_attestation_id(attestation_id)
    row = await svc.patch_attestation(
        session,
        attestation_id=uid,
        user_id=auth.user_id,
        action=payload.action,
        organization=payload.organization,
        scope_subject=payload.scope_subject,
        standards_requested=payload.standards_requested,
    )
    await session.commit()
    artifact_count = await svc.count_artifacts(
        session, attestation_id=row.id
    )
    return _attestation_to_info(row, artifact_count=artifact_count)


# ── POST /v1/attestations/{id}/artifacts ───────────────────────────


@router.post(
    "/attestations/{attestation_id}/artifacts",
    response_model=AttestationArtifactInfo,
    status_code=201,
)
async def upload_artifact(
    attestation_id: str,
    payload: AttestationArtifactRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AttestationArtifactInfo:
    """Upload a supporting evidence artifact. The actual file bytes
    travel as base64-encoded ``content_b64``; v0.0.1 caps at 50 MB
    decoded."""
    uid = parse_attestation_id(attestation_id)
    outcome = await svc.upload_artifact(
        session,
        attestation_id=uid,
        user_id=auth.user_id,
        kind=payload.kind,
        filename=payload.filename,
        content_b64=payload.content_b64,
        encrypted=payload.encrypted,
    )
    await session.commit()
    return _artifact_to_info(outcome.artifact)


# ── POST /v1/attestations/{id}/renew ───────────────────────────────


@router.post(
    "/attestations/{attestation_id}/renew",
    response_model=AttestationRenewalInfo,
    status_code=201,
)
async def renew_attestation(
    attestation_id: str,
    payload: AttestationRenewalRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AttestationRenewalInfo:
    """Initiate a renewal cycle. Idempotent on
    ``idempotency_key`` within the standard 24 h window."""
    uid = parse_attestation_id(attestation_id)
    renewal = await svc.initiate_renewal(
        session,
        attestation_id=uid,
        user_id=auth.user_id,
        idempotency_key=payload.idempotency_key,
    )
    await session.commit()
    return _renewal_to_info(renewal)


# ── DELETE /v1/attestations/{id} ───────────────────────────────────


@router.delete(
    "/attestations/{attestation_id}", response_model=AttestationInfo
)
async def revoke_attestation(
    attestation_id: str,
    payload: AttestationRevokeRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AttestationInfo:
    """Customer-initiated revocation. Multi-party revocation under
    §5 D12 condition 1 (material misrepresentation) is reserved for
    the audit-trail surface in 31.E.

    DELETE accepts a JSON body so the customer can record a reason
    for revocation; FastAPI supports request bodies on DELETE via
    the explicit Pydantic model annotation.
    """
    uid = parse_attestation_id(attestation_id)
    row = await svc.revoke_attestation(
        session,
        attestation_id=uid,
        user_id=auth.user_id,
        reason=payload.revocation_reason,
    )
    await session.commit()
    artifact_count = await svc.count_artifacts(
        session, attestation_id=row.id
    )
    return _attestation_to_info(row, artifact_count=artifact_count)


__all__ = ["router"]
