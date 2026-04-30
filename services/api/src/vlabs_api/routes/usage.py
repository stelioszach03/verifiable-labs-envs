"""``GET /v1/usage`` — current month's traces vs tier quota."""
from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.db import get_db
from vlabs_api.schemas import (
    TierQuota,
    UsageCounts,
    UsagePeriod,
    UsageRemaining,
    UsageResponse,
)
from vlabs_api.usage import get_current_counter, quota_remaining, tier_limits

router = APIRouter(tags=["billing"])


def _month_bounds() -> UsagePeriod:
    today = datetime.now(UTC).date()
    start = date(today.year, today.month, 1)
    if today.month == 12:
        next_start = date(today.year + 1, 1, 1)
    else:
        next_start = date(today.year, today.month + 1, 1)
    return UsagePeriod(start=start.isoformat(), end=next_start.isoformat())


@router.get("/usage", response_model=UsageResponse)
async def usage_endpoint(
    auth: AuthContext = Depends(require_api_key),
    session: AsyncSession = Depends(get_db),
) -> UsageResponse:
    counter = await get_current_counter(session, auth.api_key_id)
    cap, rpm = tier_limits(auth.tier)
    used_traces = counter.traces_count if counter else 0
    return UsageResponse(
        tier=auth.tier,
        quota=TierQuota(traces_per_month=cap, rpm=rpm),
        current_period=_month_bounds(),
        usage=UsageCounts(
            traces=used_traces,
            calibrations=counter.calibrations_count if counter else 0,
            evaluations=counter.evaluations_count if counter else 0,
            predictions=counter.predictions_count if counter else 0,
        ),
        remaining=UsageRemaining(traces=quota_remaining(auth.tier, used_traces)),
    )
