"""``GET /v1/audit/{calibration_id}`` — calibration metadata + eval history."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import CalibrationRun, Evaluation, get_db
from vlabs_api.errors import CalibrationNotFound
from vlabs_api.ids import encode_calibration_id, parse_calibration_id
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import AuditEvaluation, AuditResponse

router = APIRouter(tags=["audit"])


@router.get("/audit/{calibration_id}", response_model=AuditResponse)
async def audit_endpoint(
    calibration_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> AuditResponse:
    calib_uuid = parse_calibration_id(calibration_id)
    res = await session.execute(
        select(CalibrationRun).where(
            CalibrationRun.id == calib_uuid,
            CalibrationRun.api_key_id == auth.api_key_id,
        )
    )
    run = res.scalar_one_or_none()
    if run is None:
        raise CalibrationNotFound(detail=f"calibration_id={calibration_id}")

    res_evals = await session.execute(
        select(Evaluation)
        .where(Evaluation.calibration_id == run.id)
        .order_by(Evaluation.created_at.desc())
        .limit(100)
    )
    evals = res_evals.scalars().all()

    return AuditResponse(
        calibration_id=encode_calibration_id(run.id),
        created_at=run.created_at,
        alpha=run.alpha,
        nonconformity=run.nonconformity,  # type: ignore[arg-type]
        n_calibration=run.n_calibration,
        quantile=run.quantile,
        target_coverage=1.0 - run.alpha,
        nonconformity_stats=dict(run.nonconformity_stats),
        metadata=dict(run.extra_metadata),
        evaluations=[
            AuditEvaluation(
                n=e.n,
                empirical_coverage=e.empirical_coverage,
                passes=e.passes,
                ts=e.created_at,
            )
            for e in evals
        ],
    )
