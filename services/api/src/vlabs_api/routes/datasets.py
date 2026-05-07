"""``/v1/datasets`` — async synthetic-dataset jobs.

- ``POST /v1/datasets`` (Phase 23.B): validates the request, encrypts
  the customer's LLM API key, persists the ``dataset_jobs`` row in
  state ``queued``, and enqueues the job on the Redis worker queue.
- ``GET /v1/datasets`` (Phase 23.D): paginated list of the caller's
  jobs, optionally filtered by state.
- ``GET /v1/datasets/{dataset_id}`` (Phase 23.D): single job detail
  including aggregate reward stats once the job has succeeded.
- ``GET /v1/datasets/{dataset_id}/download`` (Phase 23.D): 302 redirect
  to a presigned R2 URL by default; ``Accept: application/json`` returns
  the URL inline alongside the SHA-256 + size.

PHASE_23_PLAN.md §5.D6 + §11: ``X-Idempotency-Key`` header dedups job
creation within a 24 h window. In-window re-issues return the original
``dataset_id`` + current state without consuming quota.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import DatasetJob, get_db
from vlabs_api.errors import (
    DatasetJobInvalidState,
    DatasetJobNotFound,
    QuotaExceeded,
    UnknownEnvironment,
)
from vlabs_api.idempotency import find_idempotent_row, is_within_window
from vlabs_api.ids import encode_dataset_id, parse_dataset_id
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import (
    DatasetCreateRequest,
    DatasetCreateResponse,
    DatasetDownloadResponse,
    DatasetJobList,
    DatasetJobResponse,
    DatasetJobSummary,
)
from vlabs_api.storage import generate_signed_url
from vlabs_api.usage import (
    get_current_counter,
    tier_tuples_limit,
)

router = APIRouter(tags=["datasets"])

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def _validate_env_id(env_id: str) -> None:
    """Lift to ``UnknownEnvironment`` if env_id is not registered.

    Lazy import keeps the FastAPI startup cold path fast (avoids
    loading numpy/sympy on every endpoint registration).
    """
    from verifiable_labs_envs import list_environments

    if env_id not in list_environments():
        raise UnknownEnvironment(detail=f"env_id={env_id!r}")


def _job_to_response(job: DatasetJob, env_version: str) -> DatasetCreateResponse:
    return DatasetCreateResponse(
        dataset_id=encode_dataset_id(job.id),
        state=job.state,  # type: ignore[arg-type]
        requested_tuples=job.requested_tuples,
        seed_start=int(job.seed_start),
        seed_end=int(job.seed_end),
        output_format=job.output_format,  # type: ignore[arg-type]
        env_version=env_version,
        created_at=job.created_at,
    )


def _job_to_full_response(job: DatasetJob) -> DatasetJobResponse:
    """Phase 23.D — full status detail. The customer's LLM API key is
    NEVER returned (only the URL + model)."""
    return DatasetJobResponse(
        dataset_id=encode_dataset_id(job.id),
        env_id=job.env_id,
        env_version=job.env_version,
        requested_tuples=int(job.requested_tuples),
        generated_tuples=int(job.generated_tuples),
        seed_start=int(job.seed_start),
        seed_end=int(job.seed_end),
        llm_endpoint_url=job.llm_endpoint_url,
        llm_model=job.llm_model,
        output_format=job.output_format,  # type: ignore[arg-type]
        budget_usd_cap=(
            float(job.budget_usd_cap) if job.budget_usd_cap is not None else None
        ),
        budget_usd_spent=float(job.budget_usd_spent or 0.0),
        state=job.state,  # type: ignore[arg-type]
        mean_reward=(float(job.mean_reward) if job.mean_reward is not None else None),
        std_reward=(float(job.std_reward) if job.std_reward is not None else None),
        p25_reward=(float(job.p25_reward) if job.p25_reward is not None else None),
        p50_reward=(float(job.p50_reward) if job.p50_reward is not None else None),
        p75_reward=(float(job.p75_reward) if job.p75_reward is not None else None),
        completion_success_rate=(
            float(job.completion_success_rate)
            if job.completion_success_rate is not None
            else None
        ),
        storage_sha256=job.storage_sha256,
        storage_size_bytes=(
            int(job.storage_size_bytes) if job.storage_size_bytes is not None else None
        ),
        error=job.error,
        idempotency_key=job.idempotency_key,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _job_to_summary(job: DatasetJob) -> DatasetJobSummary:
    return DatasetJobSummary(
        dataset_id=encode_dataset_id(job.id),
        env_id=job.env_id,
        env_version=job.env_version,
        requested_tuples=int(job.requested_tuples),
        generated_tuples=int(job.generated_tuples),
        state=job.state,  # type: ignore[arg-type]
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.post(
    "/datasets",
    response_model=DatasetCreateResponse,
    status_code=201,
)
async def create_dataset_job(
    payload: DatasetCreateRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> DatasetCreateResponse:
    from verifiable_labs_envs import __version__ as env_version

    # 1. Idempotency hit — return cached row if within window.
    cached = await find_idempotent_row(
        session, DatasetJob, auth.user_id, payload.idempotency_key
    )
    if cached is not None:
        if is_within_window(cached):
            return _job_to_response(cached, env_version=cached.env_version)
        # Out of window: clear the stale row so the partial unique
        # index doesn't block the fresh insert.
        await session.delete(cached)
        await session.flush()

    # 2. Validate env_id against the registry.
    _validate_env_id(payload.env_id)

    # 3. Quota pre-flight — the request can't fit if the user has
    # already consumed (cap - requested_tuples + 1) tuples this month.
    counter = await get_current_counter(session, auth.api_key_id)
    used = counter.tuples_generated if counter else 0
    cap = tier_tuples_limit(auth.tier)
    if used + payload.requested_tuples > cap:
        raise QuotaExceeded(
            detail=(
                f"tier={auth.tier} tuples_cap={cap}, used={used}, "
                f"requested={payload.requested_tuples}; upgrade or wait "
                "for next month"
            )
        )

    # 4. Encrypt the customer's LLM API key + persist the job row.
    seed_end = int(payload.seed_start) + int(payload.requested_tuples) - 1
    job = DatasetJob(
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        env_id=payload.env_id,
        env_version=env_version,
        requested_tuples=int(payload.requested_tuples),
        seed_start=int(payload.seed_start),
        seed_end=seed_end,
        llm_endpoint_url=payload.llm_endpoint_url,
        llm_api_key_encrypted=encrypt_llm_api_key(payload.llm_api_key),
        llm_model=payload.llm_model,
        output_format=payload.output_format,
        budget_usd_cap=payload.budget_usd_cap,
        idempotency_key=payload.idempotency_key,
        state="queued",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # 5. Enqueue on Redis worker queue (Phase 23.C handles dequeue).
    # Failure to enqueue is non-fatal in 23.B — the worker added in
    # 23.C will pick up any 'queued' rows on startup. Production
    # deploys MUST have Redis configured.
    try:
        from vlabs_api.redis_client import get_client

        client = get_client()
        if client is not None:
            await client.pipeline(
                ["LPUSH", "vlabs:dataset_jobs:queue", str(job.id)],
            )
    except Exception:  # noqa: BLE001
        # Log via structlog at module level if available; the worker
        # will rescue queued jobs on startup so this isn't fatal.
        import structlog
        structlog.get_logger(__name__).warning(
            "datasets.enqueue_failed", dataset_id=str(job.id)
        )

    return _job_to_response(job, env_version=env_version)


# ── Phase 23.D — read endpoints ──────────────────────────────────


@router.get("/datasets", response_model=DatasetJobList)
async def list_dataset_jobs(
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    state: str | None = Query(default=None, max_length=32),
) -> DatasetJobList:
    """Paginated list of the caller's dataset jobs.

    Sorted by ``created_at DESC`` so newest jobs land first. Optional
    ``state`` filter narrows to a single lifecycle phase
    (``queued``/``running``/``succeeded``/``failed``/...).
    """
    base = select(DatasetJob).where(DatasetJob.user_id == auth.user_id)
    count_base = select(func.count(DatasetJob.id)).where(
        DatasetJob.user_id == auth.user_id
    )
    if state is not None:
        base = base.where(DatasetJob.state == state)
        count_base = count_base.where(DatasetJob.state == state)

    total_res = await session.execute(count_base)
    total = int(total_res.scalar_one())

    res = await session.execute(
        base.order_by(DatasetJob.created_at.desc()).limit(limit).offset(offset)
    )
    rows = res.scalars().all()
    return DatasetJobList(
        items=[_job_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetJobResponse)
async def get_dataset_job(
    dataset_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> DatasetJobResponse:
    """Single dataset job detail. 404 if id is malformed, missing, or
    not owned by the authenticated user (same surface — we don't
    leak which is which)."""
    job_uuid = parse_dataset_id(dataset_id)
    res = await session.execute(
        select(DatasetJob)
        .where(DatasetJob.id == job_uuid)
        .where(DatasetJob.user_id == auth.user_id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise DatasetJobNotFound(detail=f"dataset_id={dataset_id}")
    return _job_to_full_response(row)


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
    accept: str | None = Header(default=None),
):
    """Hand out a presigned download URL for a succeeded job.

    Default response is a 302 redirect to the signed R2 URL.
    ``Accept: application/json`` returns the URL inline (preferred for
    SDKs that need to log the SHA-256 + size before downloading).
    """
    job_uuid = parse_dataset_id(dataset_id)
    res = await session.execute(
        select(DatasetJob)
        .where(DatasetJob.id == job_uuid)
        .where(DatasetJob.user_id == auth.user_id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise DatasetJobNotFound(detail=f"dataset_id={dataset_id}")
    if row.state != "succeeded" or row.storage_key is None:
        raise DatasetJobInvalidState(
            detail=f"dataset_id={dataset_id} state={row.state}; "
            "download is available only for succeeded jobs",
        )

    url, expires_at = generate_signed_url(row.storage_key)

    wants_json = accept is not None and "application/json" in accept.lower()
    if wants_json:
        return DatasetDownloadResponse(
            dataset_id=encode_dataset_id(row.id),
            download_url=url,
            expires_at=expires_at,
            sha256=row.storage_sha256 or "",
            size_bytes=int(row.storage_size_bytes or 0),
            output_format=row.output_format,  # type: ignore[arg-type]
        )
    return RedirectResponse(url=url, status_code=302)
