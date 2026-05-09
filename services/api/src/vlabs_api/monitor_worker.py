"""Monitor-run worker pool (Phase 28.C).

PHASE_28_PLAN.md §7 architecture: scheduler tick + N worker tasks
draining a Redis queue (``vlabs:monitor:queue``). Each worker:

  1. ``BRPOP`` a monitor_run_id off the queue (or wait 30 s).
  2. Mark the run ``running``; record ``started_at``.
  3. Call the customer endpoint via :func:`run_monitor_episodes` for
     each (env, episode) pair in the monitor's configuration.
  4. Compute the summary stats + verdict (regression vs baseline).
  5. Render a PDF report; upload to R2.
  6. Persist `monitor_runs` row with summary, verdict, PDF pointer.
  7. Snapshot the baseline if this is the first successful run
     (D6-A ruling).

The 28.C scope leaves **alert dispatch** to 28.D — this worker writes
the run row but does NOT invoke the email/slack channels. 28.D adds
the post-success ``schedule_alerts`` hook here.

Process model: in-app worker pool spawned at FastAPI lifespan startup
alongside the existing ``dataset_worker`` pool. Workers acquire the
existing per-env semaphore from ``vlabs_api.concurrency`` so a burst
of monitor runs doesn't starve the interactive ``/v1/score`` route.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import db as db_module
from vlabs_api.concurrency import get_semaphore
from vlabs_api.config import get_settings
from vlabs_api.db import Monitor, MonitorRun
from vlabs_api.ids import encode_monitor_id, encode_monitor_run_id
from vlabs_api.llm_key_crypto import decrypt_llm_api_key
from vlabs_api.monitor_alerts import dispatch_monitor_alerts
from vlabs_api.monitor_episode_runner import (
    compute_run_summary,
    run_monitor_episodes,
)
from vlabs_api.monitor_pdf import render_monitor_pdf
from vlabs_api.monitor_regression import compute_verdict
from vlabs_api.monitor_scheduler import scheduler_tick
from vlabs_api.redis_client import get_client as get_redis
from vlabs_api.storage import upload_dataset

log = structlog.get_logger(__name__)

QUEUE_KEY = "vlabs:monitor:queue"
DEQUEUE_TIMEOUT_S = 30
SCHEDULER_TICK_INTERVAL_S = 30


# ───────────────────────── queue ────────────────────────────────────


async def enqueue_monitor_run(run_id: uuid.UUID) -> None:
    """Push a monitor_run_id onto the Redis worker queue.

    Failure to enqueue is non-fatal: a worker that picks up a stale
    queue can also rescue ``status='queued'`` rows on startup via
    :func:`rescue_queued_runs`.
    """
    client = get_redis()
    if client is None:
        log.info("monitor_worker.no_redis", run_id=str(run_id))
        return
    try:
        await client.pipeline(["LPUSH", QUEUE_KEY, str(run_id)])
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "monitor_worker.enqueue_failed",
            run_id=str(run_id),
            error=type(exc).__name__,
        )


async def dequeue_monitor_run(
    timeout_s: int = DEQUEUE_TIMEOUT_S,
) -> uuid.UUID | None:
    client = get_redis()
    if client is None:
        await asyncio.sleep(0.05)
        return None
    try:
        result = await client.pipeline(
            ["BRPOP", QUEUE_KEY, str(timeout_s)],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("monitor_worker.dequeue_failed", error=type(exc).__name__)
        await asyncio.sleep(1.0)
        return None
    raw = result[0] if result else None
    if not raw:
        return None
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            return uuid.UUID(raw[1])
        except (ValueError, TypeError):
            return None
    return None


async def rescue_queued_runs(session: AsyncSession) -> int:
    """Re-enqueue any ``status='queued'`` rows on worker pool startup."""
    res = await session.execute(
        select(MonitorRun).where(MonitorRun.status == "queued").limit(1000)
    )
    rescued = 0
    for row in res.scalars().all():
        await enqueue_monitor_run(row.id)
        rescued += 1
    if rescued:
        log.info("monitor_worker.rescued_queued", count=rescued)
    return rescued


async def reset_stale_running(
    session: AsyncSession, *, stale_after: timedelta = timedelta(hours=1),
) -> int:
    """Mark stuck-in-running rows as failed.

    Picks up rows whose ``started_at`` is older than ``stale_after`` —
    Phase 28.C §7 restart-recovery contract. Idempotent: runs every
    pool startup.
    """
    cutoff = datetime.now(UTC) - stale_after
    res = await session.execute(
        select(MonitorRun)
        .where(MonitorRun.status == "running")
        .where(MonitorRun.started_at < cutoff)
    )
    n = 0
    for row in res.scalars().all():
        row.status = "failed"
        row.finished_at = datetime.now(UTC)
        row.error = "scheduler_lost_run"
        n += 1
    if n:
        await session.commit()
        log.info("monitor_worker.reset_stale_running", count=n)
    return n


# ───────────────────────── job processor ─────────────────────────────


async def _mark_running(session: AsyncSession, run: MonitorRun) -> None:
    run.status = "running"
    run.started_at = datetime.now(UTC)
    await session.commit()


async def _mark_failed(
    session: AsyncSession, run: MonitorRun, *, error: str,
) -> None:
    run.status = "failed"
    run.finished_at = datetime.now(UTC)
    run.error = error[:1000]
    await session.commit()


async def _mark_success(
    session: AsyncSession,
    run: MonitorRun,
    *,
    summary: dict[str, Any],
    pdf_storage_key: str,
    pdf_sha256: str,
    cost_usd_estimate: float,
    verdict: str | None = None,
    verdict_payload: dict[str, Any] | None = None,
) -> None:
    run.status = "success"
    run.finished_at = datetime.now(UTC)
    run.summary_stats = summary
    run.pdf_storage_key = pdf_storage_key
    run.pdf_sha256 = pdf_sha256
    run.cost_usd_estimate = float(cost_usd_estimate)
    if verdict is not None:
        run.regression_verdict = verdict
    if verdict_payload is not None:
        run.verdict_payload = verdict_payload
    await session.commit()


async def process_monitor_run(
    run_id: uuid.UUID,
    *,
    session_factory: Any | None = None,
    http_client: Any | None = None,
) -> None:
    """Run one monitor_runs row queued → success | failed.

    Injectable session_factory + http_client mirror the
    ``dataset_worker`` pattern so 28.C tests can pass an
    ``ASGITransport``-style fake without spinning real Redis or
    Cloudflare R2.
    """
    factory = session_factory or db_module._SessionFactory
    if factory is None:
        raise RuntimeError("DB engine not initialised")

    async with factory() as session:  # type: ignore[misc]
        res = await session.execute(
            select(MonitorRun).where(MonitorRun.id == run_id)
        )
        run = res.scalar_one_or_none()
        if run is None:
            log.warning("monitor_worker.run_not_found", run_id=str(run_id))
            return
        if run.status in ("success", "failed"):
            log.info(
                "monitor_worker.run_already_terminal",
                run_id=str(run_id),
                status=run.status,
            )
            return
        monitor_res = await session.execute(
            select(Monitor).where(Monitor.id == run.monitor_id)
        )
        monitor = monitor_res.scalar_one_or_none()
        if monitor is None:
            await _mark_failed(session, run, error="monitor_missing")
            return

        await _mark_running(session, run)

        try:
            auth_token = decrypt_llm_api_key(monitor.auth_token_encrypted)
        except Exception as exc:  # noqa: BLE001
            await _mark_failed(
                session, run, error=f"decrypt: {type(exc).__name__}",
            )
            return

        env_subset = list(monitor.env_subset)
        episodes = int(monitor.episodes_per_env)
        endpoint = monitor.model_endpoint
        model = monitor.model_name
        monitor_name = monitor.name
        monitor_id_hex = monitor.id.hex
        monitor_pk = monitor.id

    # Heavy work outside the session — don't hold the connection
    # while we make N customer HTTP calls.
    semaphores = [get_semaphore(env_id) for env_id in env_subset]
    try:
        for sem in semaphores:
            await sem.acquire()
        results = await run_monitor_episodes(
            env_subset=env_subset,
            episodes_per_env=episodes,
            endpoint_url=endpoint,
            api_key=auth_token,
            model=model,
            http_client=http_client,
        )
    except Exception as exc:  # noqa: BLE001
        async with factory() as session:  # type: ignore[misc]
            res2 = await session.execute(
                select(MonitorRun).where(MonitorRun.id == run_id)
            )
            row = res2.scalar_one()
            await _mark_failed(
                session,
                row,
                error=f"episode_loop: {type(exc).__name__}: {exc}",
            )
        return
    finally:
        for sem in semaphores:
            with contextlib.suppress(Exception):
                sem.release()

    summary = compute_run_summary(results)

    # Compute regression verdict against the (optional) baseline summary.
    baseline_summary: dict[str, Any] | None = None
    async with factory() as session:  # type: ignore[misc]
        m_res = await session.execute(
            select(Monitor).where(Monitor.id == monitor_pk)
        )
        monitor_row = m_res.scalar_one()
        baseline_run_id = monitor_row.baseline_run_id
        if baseline_run_id is not None and baseline_run_id != run_id:
            br_res = await session.execute(
                select(MonitorRun).where(MonitorRun.id == baseline_run_id)
            )
            baseline_run = br_res.scalar_one_or_none()
            if baseline_run is not None and baseline_run.summary_stats:
                baseline_summary = dict(baseline_run.summary_stats)
    verdict_payload = compute_verdict(
        current_summary=summary,
        baseline_summary=baseline_summary,
        rng_seed=int(run_id.int) & 0xFFFFFFFF,
    )
    verdict = verdict_payload["verdict"]

    # Render + upload PDF (best-effort — failure does not block the run).
    pdf_key = f"monitors/{monitor_id_hex}/{run_id.hex}.pdf"
    pdf_sha256 = ""
    try:
        pdf_bytes = render_monitor_pdf(
            monitor_name=monitor_name,
            monitor_id=encode_monitor_id(monitor_pk),
            run_id=encode_monitor_run_id(run_id),
            scheduled_at=str(run.scheduled_at if run else ""),
            finished_at=datetime.now(UTC).isoformat(),
            verdict=verdict,
            summary=summary,
        )
        # Reuse the dataset upload helper — it accepts arbitrary bytes
        # and writes under user-prefixed paths. We map (monitor_id,
        # run_id) into the same shape.
        key, sha, _size = upload_dataset(
            user_id=monitor_id_hex,
            dataset_id=run_id.hex,
            output_format="pdf",
            payload=pdf_bytes,
        )
        pdf_key = key
        pdf_sha256 = sha
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "monitor_worker.pdf_upload_failed",
            run_id=str(run_id),
            error=f"{type(exc).__name__}: {exc}",
        )

    async with factory() as session:  # type: ignore[misc]
        res2 = await session.execute(
            select(MonitorRun).where(MonitorRun.id == run_id)
        )
        run = res2.scalar_one()
        await _mark_success(
            session,
            run,
            summary=summary,
            pdf_storage_key=pdf_key,
            pdf_sha256=pdf_sha256,
            cost_usd_estimate=float(summary["cost_usd_estimate"]),
            verdict=verdict,
            verdict_payload=verdict_payload,
        )

        # D6-A: snapshot the first successful run as the baseline.
        monitor_res = await session.execute(
            select(Monitor).where(Monitor.id == monitor_pk)
        )
        monitor = monitor_res.scalar_one()
        if monitor.baseline_run_id is None:
            monitor.baseline_run_id = run.id
        monitor.last_run_at = datetime.now(UTC)
        await session.commit()

        # D7 alert dispatch — best-effort. Never blocks run-row commit
        # (which already happened above). 'ok' verdicts skip the email
        # burst entirely; warning + regressed dispatch all configured
        # channels and persist monitor_alerts rows for the audit trail.
        if verdict in ("warning", "regressed"):
            try:
                # Re-fetch monitor with alert_channels populated.
                m_res = await session.execute(
                    select(Monitor).where(Monitor.id == monitor_pk)
                )
                monitor_row = m_res.scalar_one()
                await dispatch_monitor_alerts(
                    session,
                    monitor=monitor_row,
                    run=run,
                    summary=summary,
                    verdict_payload=verdict_payload,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "monitor_worker.alert_dispatch_failed",
                    run_id=str(run_id),
                    error=f"{type(exc).__name__}: {exc}",
                )


# ───────────────────────── worker loop ─────────────────────────────


async def monitor_worker_loop(worker_id: int = 0) -> None:
    """Drain ``vlabs:monitor:queue`` indefinitely."""
    log.info("monitor_worker.loop_started", worker_id=worker_id)
    while True:
        try:
            run_id = await dequeue_monitor_run()
            if run_id is None:
                continue
            try:
                await process_monitor_run(run_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "monitor_worker.process_failed",
                    run_id=str(run_id),
                    error=f"{type(exc).__name__}: {exc}",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "monitor_worker.loop_unhandled",
                error=type(exc).__name__,
            )
            await asyncio.sleep(1.0)


async def scheduler_loop(
    *,
    interval_s: float = SCHEDULER_TICK_INTERVAL_S,
    session_factory: Any | None = None,
) -> None:
    """Tick the monitor scheduler at fixed intervals."""
    factory = session_factory or db_module._SessionFactory
    if factory is None:
        log.warning("monitor_scheduler.no_session_factory")
        return
    log.info("monitor_scheduler.loop_started", interval_s=interval_s)
    while True:
        try:
            async with factory() as session:  # type: ignore[misc]
                enqueued_ids = await scheduler_tick(session)
            for rid in enqueued_ids:
                await enqueue_monitor_run(rid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "monitor_scheduler.tick_failed",
                error=type(exc).__name__,
            )
        await asyncio.sleep(interval_s)


async def spawn_monitor_pool() -> list[asyncio.Task]:
    """Spawn N monitor workers + 1 scheduler tick task at lifespan start."""
    settings = get_settings()
    pool_size = max(1, int(settings.vlabs_data_worker_pool_size))
    tasks: list[asyncio.Task] = []
    # Restart recovery: reset stale 'running' rows + re-enqueue queued ones.
    factory = db_module._SessionFactory
    if factory is not None:
        async with factory() as session:  # type: ignore[misc]
            await reset_stale_running(session)
            await rescue_queued_runs(session)
    # Spawn worker pool.
    for i in range(pool_size):
        tasks.append(
            asyncio.create_task(monitor_worker_loop(worker_id=i))
        )
    # Spawn the scheduler tick — single instance per machine.
    tasks.append(asyncio.create_task(scheduler_loop()))
    return tasks


__all__ = [
    "QUEUE_KEY",
    "enqueue_monitor_run",
    "dequeue_monitor_run",
    "rescue_queued_runs",
    "reset_stale_running",
    "process_monitor_run",
    "monitor_worker_loop",
    "scheduler_loop",
    "spawn_monitor_pool",
]
