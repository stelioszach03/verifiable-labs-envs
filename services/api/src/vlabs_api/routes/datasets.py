"""``POST /v1/datasets`` — create an async synthetic-dataset job (Phase 23.B).

Validates the request, encrypts the customer's LLM API key, persists
the ``dataset_jobs`` row in state ``queued``, enqueues the job_id on
the Redis worker queue, and returns immediately. The actual generation
runs asynchronously via the worker pool added in Phase 23.C; clients
poll ``GET /v1/datasets/{dataset_id}`` (Phase 23.D) for status.

PHASE_23_PLAN.md §5.D6 + §11: ``X-Idempotency-Key`` header dedups job
creation within a 24 h window. In-window re-issues return the original
``dataset_id`` + current state without consuming quota.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import DatasetJob, get_db
from vlabs_api.errors import QuotaExceeded, UnknownEnvironment
from vlabs_api.idempotency import find_idempotent_row, is_within_window
from vlabs_api.ids import encode_dataset_id
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import DatasetCreateRequest, DatasetCreateResponse
from vlabs_api.usage import (
    get_current_counter,
    tier_tuples_limit,
)

router = APIRouter(tags=["datasets"])


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
