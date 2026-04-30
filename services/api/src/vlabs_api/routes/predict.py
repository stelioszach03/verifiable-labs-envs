"""``POST /v1/predict`` — return the conformal interval for one prediction.

Hot path in production. Per Stelios's Q5 answer: 1 predict call = 1
billable trace.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.calibration import predict_interval
from vlabs_api.db import CalibrationRun, get_db
from vlabs_api.errors import CalibrationNotFound, QuotaExceeded
from vlabs_api.ids import encode_calibration_id, parse_calibration_id
from vlabs_api.ratelimit import DEFAULT_LIMIT, limiter
from vlabs_api.schemas import PredictRequest, PredictResponse
from vlabs_api.usage import (
    get_current_counter,
    increment_counter,
    tier_limits,
)

router = APIRouter(tags=["calibration"])


@router.post("/predict", response_model=PredictResponse)
@limiter.limit(DEFAULT_LIMIT)
async def predict_endpoint(
    request: Request,
    payload: PredictRequest,
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_db),
) -> PredictResponse:
    calib_uuid = parse_calibration_id(payload.calibration_id)
    res = await session.execute(
        select(CalibrationRun).where(
            CalibrationRun.id == calib_uuid,
            CalibrationRun.api_key_id == auth.api_key_id,
        )
    )
    run = res.scalar_one_or_none()
    if run is None:
        raise CalibrationNotFound(detail=f"calibration_id={payload.calibration_id}")

    counter = await get_current_counter(session, auth.api_key_id)
    used = counter.traces_count if counter else 0
    cap, _ = tier_limits(auth.tier)
    if used + 1 > cap:
        raise QuotaExceeded(detail=f"tier={auth.tier} cap={cap}, used={used}")

    lower, upper, sigma_used = predict_interval(
        run.nonconformity,
        run.quantile,
        payload.predicted_reward,
        payload.uncertainty,
    )

    await increment_counter(
        session,
        auth.api_key_id,
        traces=1,
        predictions=1,
    )
    await session.commit()

    return PredictResponse(
        calibration_id=encode_calibration_id(run.id),
        predicted_reward=float(payload.predicted_reward),
        sigma=sigma_used,
        interval=(lower, upper),
        quantile=run.quantile,
        alpha=run.alpha,
        target_coverage=1.0 - run.alpha,
    )
