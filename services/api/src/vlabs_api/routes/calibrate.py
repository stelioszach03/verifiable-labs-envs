"""``POST /v1/calibrate`` — fit a conformal calibration on uploaded triples."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.calibration import calibrate_from_triples
from vlabs_api.db import CalibrationRun, get_db
from vlabs_api.errors import QuotaExceeded
from vlabs_api.ids import encode_calibration_id
from vlabs_api.ratelimit import DEFAULT_LIMIT, limiter
from vlabs_api.schemas import CalibrateRequest, CalibrateResponse
from vlabs_api.usage import (
    get_current_counter,
    increment_counter,
    tier_limits,
)

router = APIRouter(tags=["calibration"])


@router.post("/calibrate", response_model=CalibrateResponse)
@limiter.limit(DEFAULT_LIMIT)
async def calibrate_endpoint(
    request: Request,
    payload: CalibrateRequest,
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_db),
) -> CalibrateResponse:
    counter = await get_current_counter(session, auth.api_key_id)
    used = counter.traces_count if counter else 0
    cap, _ = tier_limits(auth.tier)
    n = len(payload.traces)
    if used + n > cap:
        raise QuotaExceeded(
            detail=(
                f"tier={auth.tier} cap={cap}, used={used}, requested={n}; "
                f"upgrade or wait for next month"
            )
        )

    outcome = calibrate_from_triples(
        payload.traces, alpha=payload.alpha, nonconformity_name=payload.nonconformity
    )

    run = CalibrationRun(
        api_key_id=auth.api_key_id,
        alpha=payload.alpha,
        nonconformity=payload.nonconformity,
        n_calibration=n,
        quantile=outcome.quantile,
        nonconformity_stats=outcome.nonconformity_stats,
        extra_metadata=payload.metadata,
        request_bytes=0,
        request_traces=n,
    )
    session.add(run)
    await session.flush()

    await increment_counter(
        session,
        auth.api_key_id,
        traces=n,
        calibrations=1,
    )
    await session.commit()
    await session.refresh(run)

    return CalibrateResponse(
        calibration_id=encode_calibration_id(run.id),
        alpha=run.alpha,
        nonconformity=run.nonconformity,  # type: ignore[arg-type]
        n_calibration=run.n_calibration,
        quantile=run.quantile,
        target_coverage=1.0 - run.alpha,
        nonconformity_stats=dict(run.nonconformity_stats),
        created_at=run.created_at,
    )
