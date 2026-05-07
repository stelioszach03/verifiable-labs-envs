"""Generalised idempotency-row lookup (Phase 23.B).

Phase 22.C introduced ``find_idempotent_audit`` for the
``audit_calls`` table; Phase 23.B reuses the same query shape for
``dataset_jobs``. Rather than duplicate the logic per table, this
module provides a single helper that takes the ORM table class +
user_id + idempotency key and returns the matching row (if any).

Window check is separate (:func:`is_within_idempotency_window`) so
callers can implement the spec §11 "stale row → delete-and-replace"
pattern: the partial unique index on
``(idempotency_key, user_id) WHERE idempotency_key IS NOT NULL``
blocks two non-NULL rows per (key, user), so the route handler must
delete the stale row before inserting a fresh one.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Default 24h dedup window — matches PHASE_22_PLAN.md §5.5 +
# PHASE_23_PLAN.md §11.
DEFAULT_WINDOW = timedelta(hours=24)


async def find_idempotent_row(
    session: AsyncSession,
    model: Any,
    user_id: uuid.UUID,
    idempotency_key: str | None,
) -> Any | None:
    """Return the most recent row matching ``(idempotency_key, user_id)``.

    Window check is NOT applied here — caller must invoke
    :func:`is_within_window` on the returned row to decide between
    "return cached" (in window) and "delete-and-replace" (out of
    window).
    """
    if not idempotency_key:
        return None
    res = await session.execute(
        select(model)
        .where(model.user_id == user_id)
        .where(model.idempotency_key == idempotency_key)
        .order_by(model.created_at.desc())
        .limit(1)
    )
    return res.scalar_one_or_none()


def is_within_window(row: Any, window: timedelta = DEFAULT_WINDOW) -> bool:
    """True iff ``row.created_at`` is within ``window`` of now (UTC)."""
    cutoff = datetime.now(UTC) - window
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at >= cutoff


__all__ = [
    "DEFAULT_WINDOW",
    "find_idempotent_row",
    "is_within_window",
]
