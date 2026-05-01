"""``GET /v1/admin/dashboard`` — Clerk-authed, admin-allowlisted aggregate stats.

Authorization is keyed on the comma-separated allowlist
``VLABS_ADMIN_CLERK_IDS``. A Clerk session token from a user **not**
in the list returns ``403 not_admin``.

Returns row counts plus the most recent calibration runs (no body
content is leaked across tenants — only counts and a redacted
``api_key_prefix``).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.clerk_auth import require_clerk_user
from vlabs_api.config import get_settings
from vlabs_api.db import (
    APIKey,
    CalibrationRun,
    Evaluation,
    Subscription,
    User,
    get_db,
)
from vlabs_api.errors import NotAdmin
from vlabs_api.ids import encode_calibration_id
from vlabs_api.schemas import (
    AdminDashboardCounts,
    AdminDashboardLastRun,
    AdminDashboardResponse,
)

router = APIRouter(tags=["admin"])


def _admin_allowlist() -> set[str]:
    raw = get_settings().vlabs_admin_clerk_ids or ""
    return {part.strip() for part in raw.split(",") if part.strip()}


async def require_admin(user: User = Depends(require_clerk_user)) -> User:
    if not user.clerk_user_id or user.clerk_user_id not in _admin_allowlist():
        raise NotAdmin()
    return user


@router.get("/admin/dashboard", response_model=AdminDashboardResponse)
async def admin_dashboard(
    _admin: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db),
) -> AdminDashboardResponse:
    n_users = (await session.execute(select(User).with_only_columns(User.id))).all()
    n_active = (
        await session.execute(
            select(APIKey.id).where(APIKey.revoked_at.is_(None))
        )
    ).all()
    n_revoked = (
        await session.execute(
            select(APIKey.id).where(APIKey.revoked_at.is_not(None))
        )
    ).all()
    n_calib = (
        await session.execute(select(CalibrationRun.id))
    ).all()
    n_eval = (await session.execute(select(Evaluation.id))).all()
    n_subs = (
        await session.execute(
            select(Subscription.id).where(
                Subscription.status.in_(("active", "trialing"))
            )
        )
    ).all()

    recent_q = await session.execute(
        select(CalibrationRun, APIKey.key_prefix)
        .join(APIKey, CalibrationRun.api_key_id == APIKey.id)
        .order_by(CalibrationRun.created_at.desc())
        .limit(10)
    )
    recent: list[AdminDashboardLastRun] = []
    for run, prefix in recent_q.all():
        recent.append(
            AdminDashboardLastRun(
                calibration_id=encode_calibration_id(run.id),
                api_key_prefix=prefix,
                n_calibration=run.n_calibration,
                quantile=run.quantile,
                created_at=run.created_at,
            )
        )

    return AdminDashboardResponse(
        counts=AdminDashboardCounts(
            users=len(n_users),
            api_keys_active=len(n_active),
            api_keys_revoked=len(n_revoked),
            calibrations_total=len(n_calib),
            evaluations_total=len(n_eval),
            subscriptions_active=len(n_subs),
        ),
        most_recent_calibrations=recent,
        billing_enabled=get_settings().vlabs_billing_enabled,
    )
