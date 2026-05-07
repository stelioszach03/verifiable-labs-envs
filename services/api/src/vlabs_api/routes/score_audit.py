"""``GET /v1/score/audit`` + ``GET /v1/score/audit/{audit_id}`` (Phase 22.D).

Two endpoints, both ``X-Vlabs-Key`` (data plane) authed:

- ``GET /v1/score/audit/{audit_id}`` — single audit call detail by id.
- ``GET /v1/score/audit?limit=N&offset=K`` — paginated list per user.

PHASE_22_PLAN.md §5.10 namespaces these under ``/v1/score/audit/`` to
avoid the path-template collision with the existing Phase 16
calibration audit at ``/v1/audit/{calibration_id}``.

Per spec §5.3 the completion text is never persisted; ``completion_hash``
is the SHA-256 hex string. Customers verify a row matches their
completion by re-hashing locally — nobody else can recover the text.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import AuditCall, get_db
from vlabs_api.errors import AuditCallNotFound
from vlabs_api.ids import encode_audit_id, parse_audit_id
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import AuditCallList, AuditCallResponse, AuditCallSummary

router = APIRouter(tags=["training"])

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def _row_to_response(row: AuditCall) -> AuditCallResponse:
    return AuditCallResponse(
        audit_id=encode_audit_id(row.id),
        env_id=row.env_id,
        env_version=row.env_version,
        seed=int(row.seed),
        completion_hash=row.completion_hash,
        reward=float(row.reward),
        conformal_interval=(float(row.conformal_low), float(row.conformal_high)),
        coverage_guarantee=float(row.coverage),
        components_breakdown={k: float(v) for k, v in row.components_json.items()},
        latency_ms=int(row.latency_ms),
        idempotency_key=row.idempotency_key,
        created_at=row.created_at,
    )


def _row_to_summary(row: AuditCall) -> AuditCallSummary:
    return AuditCallSummary(
        audit_id=encode_audit_id(row.id),
        env_id=row.env_id,
        env_version=row.env_version,
        reward=float(row.reward),
        latency_ms=int(row.latency_ms),
        created_at=row.created_at,
    )


@router.get("/score/audit/{audit_id}", response_model=AuditCallResponse)
async def get_audit_call(
    audit_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AuditCallResponse:
    """Single audit-call detail. 404 if id is malformed, missing, or
    not owned by the authenticated user (RLS-safe — same surface)."""
    audit_uuid = parse_audit_id(audit_id)
    res = await session.execute(
        select(AuditCall)
        .where(AuditCall.id == audit_uuid)
        .where(AuditCall.user_id == auth.user_id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise AuditCallNotFound(detail=f"audit_id={audit_id}")
    return _row_to_response(row)


@router.get("/score/audit", response_model=AuditCallList)
async def list_audit_calls(
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> AuditCallList:
    """Paginated list of audit calls owned by the authenticated user.

    Sorted by ``created_at DESC`` so the newest calls land first.
    Offset-pagination keeps the contract simple; if a user has > 1 M
    rows we'll switch to cursor pagination in a follow-up.
    """
    total_res = await session.execute(
        select(func.count(AuditCall.id)).where(AuditCall.user_id == auth.user_id)
    )
    total = int(total_res.scalar_one())

    res = await session.execute(
        select(AuditCall)
        .where(AuditCall.user_id == auth.user_id)
        .order_by(AuditCall.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = res.scalars().all()

    return AuditCallList(
        items=[_row_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
