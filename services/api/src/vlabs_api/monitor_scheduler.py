"""Scheduler tick for continuous-capability monitors (Phase 28.C).

PHASE_28_PLAN.md §7 D1-D ruling: ``monitors.next_run_at`` is the
single source of truth. The tick reads ``monitors WHERE status='active'
AND next_run_at <= now()``, creates ``monitor_runs`` rows for each,
advances ``next_run_at`` per the cadence, and returns the new run IDs
(the worker module enqueues them on Redis).

``SELECT … FOR UPDATE SKIP LOCKED`` makes the tick safe across two
machines firing at the same wall-clock second; the
``UNIQUE(monitor_id, scheduled_at)`` constraint on ``monitor_runs``
provides the second line of defence (R10 mitigation).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.monitor_cadence import compute_next_run_at

log = structlog.get_logger(__name__)


async def _create_monitor_run(
    session: AsyncSession,
    monitor: Monitor,
    *,
    scheduled_at: datetime,
    trigger: str = "scheduled",
) -> MonitorRun:
    """INSERT a fresh monitor_runs row in status='queued'.

    Race-tolerant via the ``monitor_runs_idempotency`` UNIQUE
    constraint on ``(monitor_id, scheduled_at)`` — a duplicate insert
    raises :class:`IntegrityError` which the caller treats as
    "another tick already enqueued this run".
    """
    run = MonitorRun(
        monitor_id=monitor.id,
        scheduled_at=scheduled_at,
        status="queued",
        trigger=trigger,
    )
    session.add(run)
    await session.flush()
    return run


async def scheduler_tick(session: AsyncSession) -> list[uuid.UUID]:
    """One pass: enqueue any due monitors. Returns new run IDs.

    Returns the list rather than directly LPUSHing on Redis so the
    caller (`vlabs_api.monitor_worker.scheduler_loop`) can perform
    the queue write outside the DB transaction window — keeps the
    Postgres connection released as quickly as possible.
    """
    now = datetime.now(UTC)
    res = await session.execute(
        select(Monitor)
        .where(Monitor.status == "active")
        .where(Monitor.next_run_at <= now)
        .with_for_update(skip_locked=True)
    )
    enqueued_ids: list[uuid.UUID] = []
    for monitor in res.scalars().all():
        scheduled_at = monitor.next_run_at
        try:
            run = await _create_monitor_run(
                session, monitor,
                scheduled_at=scheduled_at, trigger="scheduled",
            )
        except IntegrityError:
            # Duplicate — another tick already created the run.
            await session.rollback()
            continue
        # Advance next_run_at past now so the next tick picks the
        # *future* slot, not the missed one (catch-up is single-shot).
        monitor.last_run_at = now
        monitor.next_run_at = compute_next_run_at(
            monitor.cadence,  # type: ignore[arg-type]
            anchor=now,
        )
        enqueued_ids.append(run.id)

    if enqueued_ids:
        await session.commit()
    return enqueued_ids


async def schedule_manual_run(
    session: AsyncSession, monitor: Monitor,
) -> MonitorRun:
    """Create an ad-hoc 'manual' monitor run.

    Triggered by ``POST /v1/monitors/{id}/run``. Does NOT advance
    ``next_run_at`` — the next scheduled run still fires on its
    original cadence.

    Multiple manual triggers in the same wall-clock microsecond would
    collide on the ``UNIQUE(monitor_id, scheduled_at)`` index; we
    nudge the timestamp by the row count to keep the trigger
    idempotent at the column level (the index doesn't care about
    sub-microsecond differences as long as the UNIQUE pair is
    distinct). This matches the dataset_jobs idempotency pattern.
    """
    now = datetime.now(UTC)
    # If a row already exists at exactly this microsecond, nudge by
    # 1µs until we land on a free slot (extremely rare in practice).
    scheduled_at = now
    for _attempt in range(5):
        try:
            run = await _create_monitor_run(
                session, monitor,
                scheduled_at=scheduled_at, trigger="manual",
            )
            await session.commit()
            return run
        except IntegrityError:
            await session.rollback()
            scheduled_at = scheduled_at.replace(
                microsecond=(scheduled_at.microsecond + 1) % 1_000_000,
            )
    raise RuntimeError(
        "could not schedule manual monitor run after 5 attempts"
    )


def _build_summary_payload(
    summary: dict[str, Any] | None,
    verdict: str | None,
    verdict_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compose the meta payload returned in the run-row JSONB column."""
    return {
        "summary": summary or {},
        "verdict": verdict,
        "verdict_payload": verdict_payload or {},
    }


__all__ = [
    "scheduler_tick",
    "schedule_manual_run",
]
