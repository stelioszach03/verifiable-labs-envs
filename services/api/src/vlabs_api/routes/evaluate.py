"""``POST /v1/evaluate`` — held-out coverage report against a calibration."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.calibration import evaluate_against_calibration
from vlabs_api.db import CalibrationRun, Evaluation, get_db
from vlabs_api.errors import CalibrationNotFound, QuotaExceeded
from vlabs_api.ids import encode_calibration_id, parse_calibration_id
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import EvaluateRequest, EvaluateResponse
from vlabs_api.usage import (
    get_current_counter,
    increment_counter,
    tier_limits,
)

router = APIRouter(tags=["calibration"])


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_endpoint(
    request: Request,
    payload: EvaluateRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> EvaluateResponse:
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
    n = len(payload.traces)
    if used + n > cap:
        raise QuotaExceeded(
            detail=f"tier={auth.tier} cap={cap}, used={used}, requested={n}"
        )

    report = evaluate_against_calibration(
        traces=payload.traces,
        nonconformity_name=run.nonconformity,
        quantile=run.quantile,
        alpha=run.alpha,
        tolerance=payload.tolerance,
    )

    eval_row = Evaluation(
        calibration_id=run.id,
        api_key_id=auth.api_key_id,
        n=int(report["n"]),
        empirical_coverage=float(report["empirical_coverage"]),
        target_coverage=float(report["target_coverage"]),
        passes=bool(report["passes"]),
        tolerance=float(report["tolerance"]),
        interval_width_mean=float(report["interval_width_mean"]),
        nonconformity_stats=dict(report["nonconformity"]),  # type: ignore[arg-type]
        request_traces=n,
    )
    session.add(eval_row)
    await session.flush()

    await increment_counter(
        session,
        auth.api_key_id,
        traces=n,
        evaluations=1,
    )
    await session.commit()

    return EvaluateResponse(
        calibration_id=encode_calibration_id(run.id),
        target_coverage=float(report["target_coverage"]),
        empirical_coverage=float(report["empirical_coverage"]),
        n=int(report["n"]),
        n_in_interval=int(report["n_in_interval"]),
        interval_width_mean=float(report["interval_width_mean"]),
        interval_width_median=float(report["interval_width_median"]),
        tolerance=float(report["tolerance"]),
        passes=bool(report["passes"]),
        nonconformity=dict(report["nonconformity"]),  # type: ignore[arg-type]
    )
