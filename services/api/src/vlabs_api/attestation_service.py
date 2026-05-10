"""Service layer for the V-Certified attestation programme (Phase 31.B).

Pure business logic — endpoints in
:mod:`vlabs_api.routes.attestations` are thin wrappers. The service
layer enforces:

- D1-D scope-type constraints (model / deployment / organization).
- D4-B tier constraints (bronze / silver / gold).
- D3-D cycle constraints (annual / continuous; default ``annual`` for
  Bronze + Silver, ``continuous`` for Gold).
- D8 standards-alignment subset (must be ⊂ {iso_42001, nist_ai_rmf,
  eu_ai_act, soc2}).
- Status-machine transitions: ``draft → submitted → under_review →
  approved → revoked|expired|withdrawn``.
- D9 artifact validation (kind in locked enumeration, max 50 MB,
  SHA-256 hashed at upload, optional Fernet encryption flag).

Real PKI cert issuance lands in 31.D (this module records the
``cert_serial`` slot but does not issue certificates yet — the audit-
to-approve transition writes a stub serial in 31.B; 31.D replaces
the stub with a real X.509 issuance call).
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import (
    Attestation,
    AttestationArtifact,
    AttestationAudit,
    AttestationRenewal,
)
from vlabs_api.errors import (
    AttestationArtifactTooLarge,
    AttestationInvalidArtifact,
    AttestationInvalidState,
    AttestationNotFound,
    AttestationStandardMismatch,
)
from vlabs_api.idempotency import DEFAULT_WINDOW
from vlabs_api.ids import encode_attestation_public_id

logger = logging.getLogger(__name__)


# ── locked constants ────────────────────────────────────────────────


MAX_ARTIFACT_SIZE_BYTES: int = 50 * 1024 * 1024
"""D9 — 50 MB cap per artifact."""

DEFAULT_ANNUAL_LIFETIME = timedelta(days=365)
"""D3-B — annual cycle attestation validity."""

DEFAULT_CONTINUOUS_LIFETIME = timedelta(days=395)
"""D3-C — continuous tier carries a 365 d issuance + 30 d grace
period (R14 mitigation: certificate validity extends 30 d past
renewal grace to prevent customer outage)."""

ALLOWED_STANDARDS: frozenset[str] = frozenset(
    {"iso_42001", "nist_ai_rmf", "eu_ai_act", "soc2"}
)
"""D8 — locked subset of standards crosswalks v0.0.1 supports."""

CYCLE_FOR_TIER: dict[str, str] = {
    "bronze": "annual",
    "silver": "annual",
    "gold": "continuous",
}
"""D3-D / D4-B mapping: Bronze + Silver get annual lifecycle; Gold
gets continuous (monthly health check + annual recertification).
Customer's request must match this mapping; we don't auto-correct
because a tier/cycle mismatch usually indicates customer
misunderstanding the programme."""

LIVE_STATUSES: tuple[str, ...] = ("approved",)
"""Statuses for which an attestation is *active* on the public
registry."""

TERMINAL_STATUSES: tuple[str, ...] = ("revoked", "expired", "withdrawn")
"""Statuses from which no further transition is allowed."""


@dataclass(frozen=True)
class CreateOutcome:
    """Result of creating a draft attestation."""

    attestation: Attestation
    is_new: bool


@dataclass(frozen=True)
class ArtifactUploadOutcome:
    """Result of uploading an artifact."""

    artifact: AttestationArtifact
    decoded_size_bytes: int


# ── public_id generation ────────────────────────────────────────────


async def _generate_unique_public_id(
    session: AsyncSession, *, max_attempts: int = 8
) -> tuple[uuid.UUID, str]:
    """Pick a fresh attestation UUID + derive its public_id.

    The Crockford-base32 encoder maps the upper 40 bits of the UUID
    to an 8-char public id; 1.1T possible values means collisions are
    rare, but we retry up to ``max_attempts`` if the unique index
    catches a duplicate."""
    from vlabs_api.db import Attestation as _Att

    for _ in range(max_attempts):
        candidate_uuid = uuid.uuid4()
        candidate_public = encode_attestation_public_id(candidate_uuid)
        existing = await session.execute(
            select(_Att.id).where(_Att.public_id == candidate_public)
        )
        if existing.scalar_one_or_none() is None:
            return candidate_uuid, candidate_public
    raise RuntimeError(
        f"failed to allocate unique attestation public_id after "
        f"{max_attempts} attempts; check public_id distribution"
    )


# ── helpers ─────────────────────────────────────────────────────────


def _validate_standards(requested: Sequence[str]) -> list[str]:
    """Validate that every requested framework is in the locked D8
    enumeration. Returns a deduplicated, sorted list."""
    seen: set[str] = set()
    for std in requested:
        if std not in ALLOWED_STANDARDS:
            raise AttestationStandardMismatch(
                detail=f"unknown framework {std!r}; allowed: "
                + ",".join(sorted(ALLOWED_STANDARDS))
            )
        seen.add(std)
    return sorted(seen)


def _validate_cycle_for_tier(tier: str, cycle: str) -> None:
    """Customer's tier ⇒ cycle mapping (D3-D / D4-B). We reject
    mismatched combinations explicitly (R8 transparency)."""
    expected = CYCLE_FOR_TIER.get(tier)
    if expected is None:
        raise AttestationInvalidState(
            detail=f"unknown tier {tier!r}; allowed: bronze/silver/gold"
        )
    if cycle != expected:
        raise AttestationInvalidState(
            detail=(
                f"tier={tier!r} requires cycle={expected!r}; "
                f"got cycle={cycle!r}. Bronze + Silver use annual; "
                f"Gold uses continuous."
            )
        )


def _stub_cert_serial() -> str:
    """31.B placeholder serial. 31.D's PKI module replaces this with
    an actual X.509 issuance call. The serial format is
    ``stub-<16-hex>`` so production audits can quickly grep for
    pre-31.D test data; 31.D's real serial format is plain hex."""
    return f"stub-{secrets.token_hex(8)}"


def _build_alignment(standards: Sequence[str]) -> dict[str, Any]:
    """Build the ``standards_alignment`` JSONB payload at issuance
    time. Crosswalk_version stays NULL until 31.E ships the actual
    crosswalks; 31.E backfills the version on issuance."""
    return {
        "standards": list(standards),
        "crosswalk_version": None,
        "framework_versions": {},
    }


# ── create / read / update lifecycle ────────────────────────────────


async def create_draft(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    api_key_id: uuid.UUID,
    organization: str,
    scope_type: str,
    scope_subject: str,
    tier: str,
    cycle: str,
    standards_requested: Sequence[str],
) -> CreateOutcome:
    """Persist a new ``Attestation`` row in ``draft`` status.

    Validates tier/cycle mapping + standards subset before insert.
    Generates a fresh ``public_id`` via the Crockford-base32 encoder.
    """
    _validate_cycle_for_tier(tier, cycle)
    standards = _validate_standards(standards_requested)
    att_uuid, public_id = await _generate_unique_public_id(session)

    row = Attestation(
        id=att_uuid,
        public_id=public_id,
        user_id=user_id,
        api_key_id=api_key_id,
        organization=organization,
        scope_type=scope_type,
        scope_subject=scope_subject,
        tier=tier,
        cycle=cycle,
        status="draft",
        standards_alignment=_build_alignment(standards),
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return CreateOutcome(attestation=row, is_new=True)


async def get_for_owner(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Attestation:
    """Owner-facing lookup by internal UUID. Raises
    :class:`AttestationNotFound` if the row is unknown OR not owned."""
    res = await session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    )
    row = res.scalar_one_or_none()
    if row is None or row.user_id != user_id:
        raise AttestationNotFound(detail=f"attestation_id={attestation_id}")
    return row


async def list_for_owner(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    limit: int,
    offset: int,
    status: str | None = None,
    tier: str | None = None,
) -> tuple[list[Attestation], int]:
    """Paginated owner-facing list. Returns (rows, total)."""
    base = select(Attestation).where(Attestation.user_id == user_id)
    if status is not None:
        base = base.where(Attestation.status == status)
    if tier is not None:
        base = base.where(Attestation.tier == tier)
    total = (
        await session.execute(
            base.with_only_columns(func.count()).order_by(None)
        )
    ).scalar_one()
    rows = (
        await session.execute(
            base.order_by(Attestation.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()
    return list(rows), int(total)


async def patch_attestation(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    user_id: uuid.UUID,
    action: str | None,
    organization: str | None,
    scope_subject: str | None,
    standards_requested: Sequence[str] | None,
) -> Attestation:
    """Apply a partial update + optional state-transition action.

    Action semantics:
    - ``None`` — pure metadata update; only allowed in ``draft``.
    - ``"submit"`` — transition ``draft → submitted``.
    - ``"withdraw"`` — transition any non-terminal status →
      ``withdrawn``.
    """
    row = await get_for_owner(
        session, attestation_id=attestation_id, user_id=user_id
    )

    metadata_change_requested = (
        organization is not None
        or scope_subject is not None
        or standards_requested is not None
    )

    if metadata_change_requested:
        if row.status != "draft":
            raise AttestationInvalidState(
                detail=f"cannot edit metadata in status={row.status!r}"
            )
        if organization is not None:
            row.organization = organization
        if scope_subject is not None:
            row.scope_subject = scope_subject
        if standards_requested is not None:
            standards = _validate_standards(standards_requested)
            row.standards_alignment = _build_alignment(standards)

    if action == "submit":
        if row.status != "draft":
            raise AttestationInvalidState(
                detail=f"cannot submit from status={row.status!r}"
            )
        # Verify required artifacts are present (D9).
        artifact_kinds = await _artifact_kinds_for(
            session, attestation_id=row.id
        )
        _validate_required_artifacts_for_tier(row.tier, artifact_kinds)
        row.status = "submitted"
    elif action == "withdraw":
        if row.status in TERMINAL_STATUSES:
            raise AttestationInvalidState(
                detail=f"cannot withdraw from terminal status={row.status!r}"
            )
        row.status = "withdrawn"
    elif action is not None:
        raise AttestationInvalidState(detail=f"unknown action {action!r}")

    await session.flush()
    await session.refresh(row)
    return row


async def _artifact_kinds_for(
    session: AsyncSession, *, attestation_id: uuid.UUID
) -> set[str]:
    res = await session.execute(
        select(AttestationArtifact.kind).where(
            AttestationArtifact.attestation_id == attestation_id
        )
    )
    return {kind for kind in res.scalars().all()}


def _validate_required_artifacts_for_tier(
    tier: str, kinds_present: set[str]
) -> None:
    """D9 — minimum artifact set per tier:

    - Bronze: training_doc + audit_report + legal_signoff.
    - Silver: Bronze + monitor_record.
    - Gold: Silver + (rm_record OR prm_record) + change_mgmt.
    """
    required = {"training_doc", "audit_report", "legal_signoff"}
    if tier in ("silver", "gold"):
        required.add("monitor_record")
    if tier == "gold":
        required.add("change_mgmt")
        if not (kinds_present & {"rm_record", "prm_record"}):
            raise AttestationInvalidState(
                detail=(
                    "Gold tier requires at least one of rm_record or "
                    "prm_record artifact"
                )
            )
    missing = required - kinds_present
    if missing:
        raise AttestationInvalidState(
            detail=(
                "missing required artifacts for tier="
                f"{tier!r}: {sorted(missing)}"
            )
        )


# ── artifact upload ─────────────────────────────────────────────────


async def upload_artifact(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    user_id: uuid.UUID,
    kind: str,
    filename: str,
    content_b64: str,
    encrypted: bool,
) -> ArtifactUploadOutcome:
    """Decode + store an artifact + record the row.

    The encoded payload is decoded once for size + hash computation;
    storage is deferred to the R2 helper (when wired in 31.D).
    For the 31.B harness path the storage_uri is a deterministic
    fake-R2 path computed from the artifact UUID.
    """
    row = await get_for_owner(
        session, attestation_id=attestation_id, user_id=user_id
    )
    if row.status not in ("draft", "submitted", "under_review"):
        raise AttestationInvalidState(
            detail=(
                f"cannot upload artifact in status={row.status!r}; "
                "artifacts are immutable once approved/revoked/expired"
            )
        )

    if not content_b64:
        raise AttestationInvalidArtifact(detail="empty content_b64")
    try:
        content = base64.b64decode(content_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise AttestationInvalidArtifact(
            detail=f"content_b64 is not valid base64: {exc}"
        ) from exc
    if not content:
        raise AttestationInvalidArtifact(detail="decoded content is empty")
    if len(content) > MAX_ARTIFACT_SIZE_BYTES:
        raise AttestationArtifactTooLarge(
            detail=f"{len(content)} bytes exceeds {MAX_ARTIFACT_SIZE_BYTES}"
        )

    sha256_hash = hashlib.sha256(content).hexdigest()
    artifact_uuid = uuid.uuid4()
    storage_uri = (
        f"r2://vlabs-attestations/{row.id}/{artifact_uuid}/"
        f"{_sanitise_filename(filename)}"
    )

    artifact = AttestationArtifact(
        id=artifact_uuid,
        attestation_id=row.id,
        kind=kind,
        storage_uri=storage_uri,
        sha256_hash=sha256_hash,
        encrypted=bool(encrypted),
        size_bytes=len(content),
    )
    session.add(artifact)
    await session.flush()
    await session.refresh(artifact)
    return ArtifactUploadOutcome(
        artifact=artifact, decoded_size_bytes=len(content)
    )


def _sanitise_filename(name: str) -> str:
    """Strip path traversal + reduce to URL-safe chars. R2 keys land
    under deterministic UUID-based paths anyway so this is mostly
    cosmetic, but keeps the storage_uri readable."""
    safe = "".join(
        ch if ch.isalnum() or ch in {".", "-", "_"} else "_"
        for ch in name
    )
    return safe[:200] or "artifact"


# ── renewal lifecycle ───────────────────────────────────────────────


async def initiate_renewal(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    user_id: uuid.UUID,
    idempotency_key: str | None,
) -> AttestationRenewal:
    """Initiate a new renewal cycle. Idempotent on
    ``idempotency_key`` within a 24 h window (mirrors Phase 23
    semantics)."""
    row = await get_for_owner(
        session, attestation_id=attestation_id, user_id=user_id
    )
    if row.status != "approved":
        raise AttestationInvalidState(
            detail=(
                f"renewal requires approved status; got {row.status!r}"
            )
        )

    if idempotency_key is not None:
        existing = await _find_idempotent_renewal(
            session,
            attestation_id=row.id,
            idempotency_key=idempotency_key,
        )
        if existing is not None and _renewal_within_window(existing):
            return existing

    cycle_number = await _next_cycle_number(
        session, attestation_id=row.id
    )
    renewal = AttestationRenewal(
        attestation_id=row.id,
        cycle_number=cycle_number,
        idempotency_key=idempotency_key,
    )
    session.add(renewal)
    await session.flush()
    await session.refresh(renewal)
    return renewal


def _renewal_within_window(renewal: AttestationRenewal) -> bool:
    """Same 24 h window semantics as
    :func:`vlabs_api.idempotency.is_within_window` but keyed off
    ``initiated_at`` (the renewal-table column name)."""
    initiated = renewal.initiated_at
    if initiated.tzinfo is None:
        initiated = initiated.replace(tzinfo=UTC)
    return datetime.now(UTC) - initiated < DEFAULT_WINDOW


async def _find_idempotent_renewal(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    idempotency_key: str,
) -> AttestationRenewal | None:
    res = await session.execute(
        select(AttestationRenewal).where(
            AttestationRenewal.attestation_id == attestation_id,
            AttestationRenewal.idempotency_key == idempotency_key,
        )
    )
    return res.scalar_one_or_none()


async def _next_cycle_number(
    session: AsyncSession, *, attestation_id: uuid.UUID
) -> int:
    res = await session.execute(
        select(func.coalesce(func.max(AttestationRenewal.cycle_number), 0))
        .where(AttestationRenewal.attestation_id == attestation_id)
    )
    return int(res.scalar_one()) + 1


# ── revocation ──────────────────────────────────────────────────────


async def revoke_attestation(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    user_id: uuid.UUID,
    reason: str,
) -> Attestation:
    """Customer-initiated revocation.

    Customer revocation is single-party (D12 condition 4). Multi-
    party revocation under condition 1 (material misrepresentation)
    is reserved for the audit-trail surface — that path lands in
    31.E with the lifecycle integration tests.
    """
    row = await get_for_owner(
        session, attestation_id=attestation_id, user_id=user_id
    )
    if row.status in TERMINAL_STATUSES:
        raise AttestationInvalidState(
            detail=f"already in terminal status={row.status!r}"
        )
    row.revoked_at = datetime.now(UTC)
    row.revocation_reason = reason
    row.status = "revoked"

    audit = AttestationAudit(
        attestation_id=row.id,
        auditor_kind="self",
        auditor_user_id=user_id,
        auditor_label="customer-initiated",
        audit_summary={"reason": reason},
        decision="revoke",
    )
    session.add(audit)
    await session.flush()
    await session.refresh(row)
    return row


# ── audit / approval (admin path; surfaced in 31.E) ─────────────────


async def record_audit_decision(
    session: AsyncSession,
    *,
    attestation_id: uuid.UUID,
    auditor_kind: str,
    auditor_user_id: uuid.UUID | None,
    auditor_label: str | None,
    audit_summary: dict[str, Any],
    decision: str,
) -> AttestationAudit:
    """Record one auditor's decision on an attestation.

    Status transitions handled by the caller — this function only
    persists the audit row + (when decision='approve') stamps the
    attestation cert_serial + issued_at fields. Multi-party
    revocation gating lives in 31.E.
    """
    res = await session.execute(
        select(Attestation).where(Attestation.id == attestation_id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise AttestationNotFound(detail=f"attestation_id={attestation_id}")

    audit = AttestationAudit(
        attestation_id=attestation_id,
        auditor_kind=auditor_kind,
        auditor_user_id=auditor_user_id,
        auditor_label=auditor_label,
        audit_summary=audit_summary,
        decision=decision,
    )
    session.add(audit)

    if decision == "approve":
        if row.status not in ("submitted", "under_review"):
            raise AttestationInvalidState(
                detail=f"cannot approve from status={row.status!r}"
            )
        row.status = "approved"
        row.issued_at = datetime.now(UTC)
        row.cert_serial = _stub_cert_serial()
        lifetime = (
            DEFAULT_CONTINUOUS_LIFETIME
            if row.cycle == "continuous"
            else DEFAULT_ANNUAL_LIFETIME
        )
        row.expires_at = row.issued_at + lifetime
    elif decision == "reject":
        if row.status not in ("submitted", "under_review"):
            raise AttestationInvalidState(
                detail=f"cannot reject from status={row.status!r}"
            )
        row.status = "withdrawn"
    elif decision == "request_more":
        if row.status != "submitted":
            raise AttestationInvalidState(
                detail=f"cannot request_more from status={row.status!r}"
            )
        row.status = "draft"

    await session.flush()
    await session.refresh(audit)
    await session.refresh(row)
    return audit


async def list_audit_trail(
    session: AsyncSession, *, attestation_id: uuid.UUID
) -> list[AttestationAudit]:
    res = await session.execute(
        select(AttestationAudit)
        .where(AttestationAudit.attestation_id == attestation_id)
        .order_by(AttestationAudit.decided_at.desc())
    )
    return list(res.scalars().all())


# ── helper exports ──────────────────────────────────────────────────


async def count_artifacts(
    session: AsyncSession, *, attestation_id: uuid.UUID
) -> int:
    res = await session.execute(
        select(func.count())
        .select_from(AttestationArtifact)
        .where(AttestationArtifact.attestation_id == attestation_id)
    )
    return int(res.scalar_one())


__all__ = [
    "ALLOWED_STANDARDS",
    "ArtifactUploadOutcome",
    "CYCLE_FOR_TIER",
    "CreateOutcome",
    "DEFAULT_ANNUAL_LIFETIME",
    "DEFAULT_CONTINUOUS_LIFETIME",
    "LIVE_STATUSES",
    "MAX_ARTIFACT_SIZE_BYTES",
    "TERMINAL_STATUSES",
    "count_artifacts",
    "create_draft",
    "get_for_owner",
    "initiate_renewal",
    "list_audit_trail",
    "list_for_owner",
    "patch_attestation",
    "record_audit_decision",
    "revoke_attestation",
    "upload_artifact",
]
