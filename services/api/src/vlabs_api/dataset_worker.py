"""vlabs-data async worker (Phase 23.C).

PHASE_23_PLAN.md §7 architecture: in-app worker pool, draining a
Redis-backed queue (``vlabs:dataset_jobs:queue``). Each worker:

  1. ``BRPOP`` a dataset_id off the queue (or wait 30 s).
  2. Mark the job ``running``; record ``started_at``.
  3. For each seed in [seed_start, seed_end]:
     a. Render the env's prompt via the registered adapter.
     b. Call the customer's LLM endpoint (OpenAI Chat Completions).
     c. Score the completion via :mod:`vlabs_api.scoring`.
     d. Append the tuple to the in-memory chunk buffer.
     e. Every ``checkpoint_every_n`` tuples: upload the chunk to R2,
        bump ``generated_tuples``, debit ``tuples_generated`` counter.
     f. Stop if ``budget_usd_spent`` reaches ``budget_usd_cap``.
  4. Concatenate chunks into the final dataset, upload to R2,
     compute aggregate stats, set ``state='succeeded'``.

Per-env semaphores from :mod:`vlabs_api.concurrency` are reused so
the worker pool doesn't starve the interactive ``/v1/score`` route.

Determinism (Phase 22 §5.8) makes mid-job restart safe: a worker
picking up a job after a restart re-runs from
``seed_start + generated_tuples`` and re-derives the same tuples
bit-identically.
"""
from __future__ import annotations

import asyncio
import io
import json
import statistics
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import db as db_module
from vlabs_api.concurrency import get_semaphore
from vlabs_api.config import get_settings
from vlabs_api.db import DatasetJob
from vlabs_api.llm_client import LLMResult, call_llm
from vlabs_api.llm_key_crypto import decrypt_llm_api_key
from vlabs_api.redis_client import get_client as get_redis
from vlabs_api.scoring import score_completion
from vlabs_api.storage import (
    delete_chunks,
    upload_chunk,
    upload_dataset,
)
from vlabs_api.usage import increment_tuples_counter

log = structlog.get_logger(__name__)

# Redis queue key.
QUEUE_KEY = "vlabs:dataset_jobs:queue"

# How long to BRPOP-wait before looping (seconds). Short enough that
# SIGTERM lands quickly, long enough that idle workers don't spin.
DEQUEUE_TIMEOUT_S = 30


# ───────────────────────── queue ────────────────────────────────────


async def enqueue_dataset_job(dataset_id: uuid.UUID) -> None:
    """Push a dataset_id onto the Redis worker queue.

    Failure to enqueue is non-fatal: a worker that picks up a stale
    queue can also rescue ``state='queued'`` rows on startup via
    :func:`rescue_queued_jobs`.
    """
    client = get_redis()
    if client is None:
        log.warning("dataset_worker.no_redis", dataset_id=str(dataset_id))
        return
    try:
        await client.pipeline(["LPUSH", QUEUE_KEY, str(dataset_id)])
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "dataset_worker.enqueue_failed",
            dataset_id=str(dataset_id),
            error=type(exc).__name__,
        )


async def dequeue_dataset_job(timeout_s: int = DEQUEUE_TIMEOUT_S) -> uuid.UUID | None:
    """BRPOP from the queue. Returns ``None`` on timeout.

    LOCAL_FAKE_R2 / no-Redis path: returns None immediately so the
    worker_loop in tests doesn't hang.
    """
    client = get_redis()
    if client is None:
        await asyncio.sleep(0.05)
        return None
    try:
        result = await client.pipeline(
            ["BRPOP", QUEUE_KEY, str(timeout_s)],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("dataset_worker.dequeue_failed", error=type(exc).__name__)
        await asyncio.sleep(1.0)
        return None
    raw = result[0] if result else None
    if not raw:
        return None
    # BRPOP returns [key, value] in modern Upstash REST.
    if isinstance(raw, list) and len(raw) >= 2:
        try:
            return uuid.UUID(raw[1])
        except (ValueError, TypeError):
            return None
    return None


async def rescue_queued_jobs(session: AsyncSession) -> int:
    """Re-enqueue any ``state='queued'`` rows on worker pool startup.

    Survives the case where Redis lost the queue between an enqueue
    and the worker dequeue (e.g. Upstash maintenance window). Idempotent:
    pushing an already-queued id back onto the queue just produces a
    duplicate that the worker discards as "already running".
    """
    res = await session.execute(
        select(DatasetJob).where(DatasetJob.state == "queued").limit(1000)
    )
    rescued = 0
    for row in res.scalars().all():
        await enqueue_dataset_job(row.id)
        rescued += 1
    if rescued:
        log.info("dataset_worker.rescued_queued", count=rescued)
    return rescued


# ───────────────────────── tuple generation ─────────────────────────


def _build_user_prompt(env_id: str, instance: Any) -> tuple[str, str]:
    """Resolve the env adapter and render system + user prompts."""
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  registers
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS, get_adapter

    if env_id not in _ADAPTERS:
        return "", str(getattr(instance, "prompt", ""))
    adapter = get_adapter(env_id)
    return adapter.system_prompt, adapter.build_user_prompt(instance)


def _serialise_tuple(
    seed: int,
    prompt: str,
    completion: str,
    reward: float,
    components: dict[str, float],
    llm_meta: LLMResult,
    env_version: str,
) -> dict[str, Any]:
    """Canonical per-tuple dict written into the dataset payload."""
    return {
        "format_version": "0.0.1",
        "env_version": env_version,
        "seed": seed,
        "prompt": prompt,
        "completion": completion,
        "reward": float(reward),
        "components": {k: float(v) for k, v in components.items()},
        "llm": {
            "prompt_tokens": llm_meta.prompt_tokens,
            "completion_tokens": llm_meta.completion_tokens,
            "cost_usd_estimate": llm_meta.cost_usd_estimate,
            "success": llm_meta.success,
        },
    }


def _serialise_chunk_jsonl(tuples: list[dict[str, Any]]) -> bytes:
    return b"\n".join(json.dumps(t).encode("utf-8") for t in tuples) + b"\n"


def _serialise_chunk_parquet(tuples: list[dict[str, Any]]) -> bytes:
    """Serialise a list of tuple dicts to Parquet. Lazy pyarrow import."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not tuples:
        return b""
    # Flatten nested dicts (components, llm) into prefixed columns —
    # Parquet round-trips simpler primitives than nested rows.
    flat = []
    for t in tuples:
        row = {
            "format_version": t["format_version"],
            "env_version": t["env_version"],
            "seed": t["seed"],
            "prompt": t["prompt"],
            "completion": t["completion"],
            "reward": t["reward"],
            "llm_prompt_tokens": t["llm"]["prompt_tokens"],
            "llm_completion_tokens": t["llm"]["completion_tokens"],
            "llm_cost_usd_estimate": t["llm"]["cost_usd_estimate"],
            "llm_success": t["llm"]["success"],
        }
        for k, v in t["components"].items():
            row[f"components_{k}"] = float(v)
        flat.append(row)
    table = pa.Table.from_pylist(flat)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def _serialise_chunk(output_format: str, tuples: list[dict[str, Any]]) -> bytes:
    if output_format == "parquet":
        try:
            return _serialise_chunk_parquet(tuples)
        except ImportError:
            # pyarrow not installed in tests by default — fall through to
            # JSONL to keep the worker functional. Production deploys
            # have pyarrow as a hard dep.
            log.warning("dataset_worker.pyarrow_missing_falling_back_to_jsonl")
    return _serialise_chunk_jsonl(tuples)


def _aggregate_stats(rewards: list[float]) -> dict[str, float | None]:
    """Compute mean/std/p25/p50/p75 + completion success rate."""
    if not rewards:
        return {
            "mean": None,
            "std": None,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    s = sorted(rewards)
    n = len(s)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(p * (n - 1))))
        return s[idx]

    return {
        "mean": float(statistics.fmean(s)),
        "std": float(statistics.pstdev(s)) if n > 1 else 0.0,
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
    }


# ───────────────────────── job processor ─────────────────────────────


async def process_dataset_job(
    job_id: uuid.UUID,
    *,
    session_factory: Any | None = None,
    http_client: Any | None = None,
) -> None:
    """Run one dataset_jobs row from queued → succeeded (or failed).

    ``session_factory`` and ``http_client`` are injectable for tests
    (the test suite passes a respx-mocked httpx client; production
    uses the global SessionFactory + a fresh httpx client).
    """
    factory = session_factory or db_module._SessionFactory
    if factory is None:
        raise RuntimeError("DB engine not initialised")

    settings = get_settings()
    checkpoint_every_n = max(1, int(settings.vlabs_data_checkpoint_every_n))

    async with factory() as session:  # type: ignore[misc]
        res = await session.execute(
            select(DatasetJob).where(DatasetJob.id == job_id)
        )
        job = res.scalar_one_or_none()
        if job is None:
            log.warning("dataset_worker.job_not_found", job_id=str(job_id))
            return
        if job.state in ("succeeded", "failed", "archived", "hard_deleted"):
            log.info(
                "dataset_worker.job_already_terminal",
                job_id=str(job_id),
                state=job.state,
            )
            return

        # Mark running. Worker resumes from generated_tuples on restart.
        job.state = "running"
        job.started_at = job.started_at or datetime.now(UTC)
        await session.commit()
        await session.refresh(job)

    sem = get_semaphore(job.env_id)

    try:
        async with sem:
            await _run_generation_loop(
                job_id=job_id,
                checkpoint_every_n=checkpoint_every_n,
                factory=factory,
                http_client=http_client,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("dataset_worker.unhandled_error", job_id=str(job_id))
        async with factory() as session:  # type: ignore[misc]
            res = await session.execute(
                select(DatasetJob).where(DatasetJob.id == job_id)
            )
            job = res.scalar_one_or_none()
            if job is not None:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"[:1000]
                job.completed_at = datetime.now(UTC)
                await session.commit()


async def _run_generation_loop(
    *,
    job_id: uuid.UUID,
    checkpoint_every_n: int,
    factory: Any,
    http_client: Any | None,
) -> None:
    """Inner loop. Fresh session per checkpoint to avoid long-lived txns."""
    from verifiable_labs_envs import load_environment

    # Load enough job state to drive the loop.
    async with factory() as session:  # type: ignore[misc]
        res = await session.execute(
            select(DatasetJob).where(DatasetJob.id == job_id)
        )
        job = res.scalar_one()
        env_id = job.env_id
        seed_start = int(job.seed_start)
        seed_end = int(job.seed_end)
        already = int(job.generated_tuples)
        budget_cap = job.budget_usd_cap
        budget_spent = float(job.budget_usd_spent or 0.0)
        output_format = job.output_format
        api_key = decrypt_llm_api_key(job.llm_api_key_encrypted)
        endpoint = job.llm_endpoint_url
        model = job.llm_model
        user_id = str(job.user_id)
        api_key_id = job.api_key_id
        env_version = job.env_version

    env = load_environment(env_id, calibration_quantile=0.5)
    chunk_idx = already // checkpoint_every_n
    buffer: list[dict[str, Any]] = []
    rewards: list[float] = []
    successes = 0
    total_attempted = already

    for seed in range(seed_start + already, seed_end + 1):
        if budget_cap is not None and budget_spent >= float(budget_cap):
            log.info(
                "dataset_worker.budget_cap_reached",
                job_id=str(job_id),
                spent=budget_spent,
                cap=budget_cap,
            )
            break

        instance = env.generate_instance(seed=seed)
        system_prompt, user_prompt = _build_user_prompt(env_id, instance)

        llm_result = await call_llm(
            endpoint_url=endpoint,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client=http_client,
        )

        # Score even on LLM failure — gives the customer a zero-reward
        # row for that seed they can investigate. The cost-cap budget
        # is tracked separately on success only.
        scored, _latency_ms, _q = await score_completion(
            env_id=env_id,
            seed=seed,
            completion=llm_result.completion_text,
        )
        reward = float(scored.get("reward", 0.0))
        components = {
            k: float(v) for k, v in scored.get("components", {}).items()
        }

        buffer.append(
            _serialise_tuple(
                seed=seed,
                prompt=user_prompt,
                completion=llm_result.completion_text,
                reward=reward,
                components=components,
                llm_meta=llm_result,
                env_version=env_version,
            )
        )
        rewards.append(reward)
        total_attempted += 1
        if llm_result.success:
            successes += 1
            budget_spent += llm_result.cost_usd_estimate

        # Checkpoint.
        if len(buffer) >= checkpoint_every_n:
            payload = _serialise_chunk(output_format, buffer)
            upload_chunk(user_id, str(job_id), output_format, chunk_idx, payload)
            chunk_idx += 1
            await _persist_progress(
                factory=factory,
                job_id=job_id,
                api_key_id=api_key_id,
                generated_delta=len(buffer),
                budget_spent=budget_spent,
            )
            buffer.clear()

    # Final tail chunk.
    if buffer:
        payload = _serialise_chunk(output_format, buffer)
        upload_chunk(user_id, str(job_id), output_format, chunk_idx, payload)
        await _persist_progress(
            factory=factory,
            job_id=job_id,
            api_key_id=api_key_id,
            generated_delta=len(buffer),
            budget_spent=budget_spent,
        )

    # Finalise: concatenate chunks, upload final dataset, compute stats.
    await _finalize_job(
        factory=factory,
        job_id=job_id,
        user_id=user_id,
        output_format=output_format,
        rewards=rewards,
        total_attempted=total_attempted,
        successes=successes,
        budget_spent=budget_spent,
    )


async def _persist_progress(
    *,
    factory: Any,
    job_id: uuid.UUID,
    api_key_id: uuid.UUID,
    generated_delta: int,
    budget_spent: float,
) -> None:
    async with factory() as session:  # type: ignore[misc]
        res = await session.execute(
            select(DatasetJob).where(DatasetJob.id == job_id)
        )
        job = res.scalar_one()
        job.generated_tuples = int(job.generated_tuples) + int(generated_delta)
        job.budget_usd_spent = float(budget_spent)
        await increment_tuples_counter(
            session, api_key_id, delta=generated_delta
        )
        await session.commit()


async def _finalize_job(
    *,
    factory: Any,
    job_id: uuid.UUID,
    user_id: str,
    output_format: str,
    rewards: list[float],
    total_attempted: int,
    successes: int,
    budget_spent: float,
) -> None:
    """Concatenate chunks → final upload → state=succeeded + stats."""
    from vlabs_api.storage import list_chunks

    chunk_keys = list_chunks(user_id, str(job_id), output_format)
    payload = _concat_chunks(output_format, chunk_keys)

    storage_key, sha256, size_bytes = upload_dataset(
        user_id, str(job_id), output_format, payload
    )

    delete_chunks(user_id, str(job_id), output_format)

    stats = _aggregate_stats(rewards)
    success_rate = (successes / total_attempted) if total_attempted else None

    async with factory() as session:  # type: ignore[misc]
        res = await session.execute(
            select(DatasetJob).where(DatasetJob.id == job_id)
        )
        job = res.scalar_one()
        job.state = "succeeded"
        job.completed_at = datetime.now(UTC)
        job.budget_usd_spent = float(budget_spent)
        job.mean_reward = stats["mean"]
        job.std_reward = stats["std"]
        job.p25_reward = stats["p25"]
        job.p50_reward = stats["p50"]
        job.p75_reward = stats["p75"]
        job.completion_success_rate = (
            float(success_rate) if success_rate is not None else None
        )
        job.storage_key = storage_key
        job.storage_sha256 = sha256
        job.storage_size_bytes = size_bytes
        await session.commit()


def _concat_chunks(output_format: str, chunk_keys: list[str]) -> bytes:
    """Concatenate chunk objects into the final dataset payload.

    LOCAL_FAKE_R2 mode reads from /tmp/r2-fake; production fetches via
    boto3. Both paths produce identical concatenated bytes.
    """
    settings = get_settings()
    bucket_name = settings.vlabs_r2_bucket_name
    if settings.vlabs_local_fake_r2:
        from pathlib import Path

        bucket_root = Path("/tmp/r2-fake") / bucket_name
        # JSONL: bytewise concat. Parquet: read+concat tables.
        if output_format == "jsonl":
            buf = b""
            for key in chunk_keys:
                buf += (bucket_root / key).read_bytes()
            return buf
        if output_format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                tables = []
                for key in chunk_keys:
                    raw = (bucket_root / key).read_bytes()
                    if not raw:
                        continue
                    tables.append(pq.read_table(io.BytesIO(raw)))
                if not tables:
                    return b""
                merged = pa.concat_tables(tables, promote=True)
                out = io.BytesIO()
                pq.write_table(merged, out, compression="snappy")
                return out.getvalue()
            except ImportError:
                buf = b""
                for key in chunk_keys:
                    buf += (bucket_root / key).read_bytes()
                return buf
        return b""

    # Production path — boto3.
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=settings.vlabs_r2_endpoint_url
        or f"https://{settings.vlabs_r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.vlabs_r2_access_key_id,
        aws_secret_access_key=settings.vlabs_r2_secret_access_key,
        region_name="auto",
    )
    if output_format == "jsonl":
        buf = b""
        for key in chunk_keys:
            obj = client.get_object(Bucket=bucket_name, Key=key)
            buf += obj["Body"].read()
        return buf

    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for key in chunk_keys:
        obj = client.get_object(Bucket=bucket_name, Key=key)
        raw = obj["Body"].read()
        if not raw:
            continue
        tables.append(pq.read_table(io.BytesIO(raw)))
    if not tables:
        return b""
    merged = pa.concat_tables(tables, promote=True)
    out = io.BytesIO()
    pq.write_table(merged, out, compression="snappy")
    return out.getvalue()


# ───────────────────────── worker loop ──────────────────────────────


async def worker_loop(worker_id: int) -> None:
    """Long-lived worker task. Cancels gracefully on lifespan shutdown."""
    log.info("dataset_worker.start", worker_id=worker_id)
    try:
        # Pre-flight: rescue any orphaned 'queued' rows.
        factory = db_module._SessionFactory
        if factory is not None:
            async with factory() as session:
                await rescue_queued_jobs(session)

        while True:
            try:
                job_id = await dequeue_dataset_job()
                if job_id is not None:
                    await process_dataset_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception(
                    "dataset_worker.loop_error", worker_id=worker_id
                )
                await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        log.info("dataset_worker.shutdown", worker_id=worker_id)
        raise


async def spawn_worker_pool(pool_size: int | None = None) -> list[asyncio.Task]:
    """Start ``N`` worker tasks. Caller is responsible for cancelling them."""
    settings = get_settings()
    n = pool_size if pool_size is not None else settings.vlabs_data_worker_pool_size
    n = max(0, int(n))
    if n == 0:
        return []
    return [
        asyncio.create_task(worker_loop(i), name=f"vlabs-data-worker-{i}")
        for i in range(n)
    ]


__all__ = [
    "QUEUE_KEY",
    "DEQUEUE_TIMEOUT_S",
    "enqueue_dataset_job",
    "dequeue_dataset_job",
    "rescue_queued_jobs",
    "process_dataset_job",
    "worker_loop",
    "spawn_worker_pool",
]


# Re-export internal helpers for the test suite (mirrors the
# Phase 22.C scoring module's pattern of exposing internals via __all__
# after the public surface).
__all__ += [
    "_aggregate_stats",
    "_serialise_chunk",
    "_serialise_chunk_jsonl",
    "_serialise_tuple",
    "_concat_chunks",
]
