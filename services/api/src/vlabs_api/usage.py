"""Usage-counter UPSERT helpers + tier resolution.

Hot-path semantics:
  * Each ``/v1/calibrate`` increments ``traces_count`` by ``n_calibration``
    and ``calibrations_count`` by 1.
  * Each ``/v1/evaluate`` increments ``traces_count`` by ``n_evaluate`` and
    ``evaluations_count`` by 1.
  * Each ``/v1/predict`` increments ``traces_count`` by 1 and
    ``predictions_count`` by 1 (per Stelios's Q5 answer: 1 predict = 1 trace).
"""
from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import TierLimits, get_settings
from vlabs_api.db import Subscription, UsageCounter

Tier = Literal["free", "pro", "team"]


def _first_day_of_month(d: date | None = None) -> date:
    today = d or datetime.now(UTC).date()
    return date(today.year, today.month, 1)


def tier_limits(tier: Tier) -> tuple[int, int]:
    """Return ``(traces_per_month, requests_per_minute)`` for a tier."""
    t: TierLimits = get_settings().tiers
    return {
        "free": (t.free_traces_per_month, t.free_rpm),
        "pro": (t.pro_traces_per_month, t.pro_rpm),
        "team": (t.team_traces_per_month, t.team_rpm),
    }[tier]


async def resolve_tier(session: AsyncSession, user_id: uuid.UUID) -> Tier:
    """Look up the user's effective tier from active subscriptions."""
    stmt = (
        select(Subscription.tier)
        .where(Subscription.user_id == user_id)
        .where(Subscription.status.in_(("active", "trialing")))
        .order_by(Subscription.current_period_end.desc())
        .limit(1)
    )
    res = await session.execute(stmt)
    row = res.scalar_one_or_none()
    if row in ("pro", "team"):
        return row  # type: ignore[return-value]
    return "free"


async def get_current_counter(
    session: AsyncSession,
    api_key_id: uuid.UUID,
    *,
    month: date | None = None,
) -> UsageCounter | None:
    m = _first_day_of_month(month)
    res = await session.execute(
        select(UsageCounter)
        .where(UsageCounter.api_key_id == api_key_id)
        .where(UsageCounter.month == m)
    )
    return res.scalar_one_or_none()


async def increment_counter(
    session: AsyncSession,
    api_key_id: uuid.UUID,
    *,
    traces: int,
    calibrations: int = 0,
    evaluations: int = 0,
    predictions: int = 0,
    month: date | None = None,
) -> None:
    """Atomic UPSERT — adds the given deltas to the (api_key_id, month) row."""
    m = _first_day_of_month(month)
    stmt = (
        pg_insert(UsageCounter)
        .values(
            api_key_id=api_key_id,
            month=m,
            traces_count=traces,
            calibrations_count=calibrations,
            evaluations_count=evaluations,
            predictions_count=predictions,
        )
        .on_conflict_do_update(
            index_elements=[UsageCounter.api_key_id, UsageCounter.month],
            set_={
                "traces_count": UsageCounter.traces_count + traces,
                "calibrations_count": UsageCounter.calibrations_count + calibrations,
                "evaluations_count": UsageCounter.evaluations_count + evaluations,
                "predictions_count": UsageCounter.predictions_count + predictions,
            },
        )
    )
    await session.execute(stmt)


def quota_remaining(tier: Tier, used_traces: int) -> int:
    """Return the number of traces still available this month."""
    cap, _ = tier_limits(tier)
    return max(cap - used_traces, 0)


__all__ = [
    "Tier",
    "tier_limits",
    "resolve_tier",
    "get_current_counter",
    "increment_counter",
    "quota_remaining",
]
