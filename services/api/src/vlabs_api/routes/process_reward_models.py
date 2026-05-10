"""``/v1/process-reward-models`` — distilled PRM service (Phase 30.E).

Five endpoints per :doc:`PHASE_30_PLAN.md` §10. All run on stub
inference until 30.G lands trained weights; the response shape is
locked NOW so frontend + SDK integrations can land in parallel.

- ``GET /v1/process-reward-models`` — paginated list with family/status filter.
- ``GET /v1/process-reward-models/{model_id}`` — single-model detail.
- ``POST /v1/process-reward-models/{model_id}/score`` — single-call
  scoring; opt-in Redis cache via ``X-Vlabs-Cache: enable`` (D10-B).
- ``POST /v1/process-reward-models/{model_id}/score/batch`` — up to
  50 (prompt, trace) pairs per call; idempotent on
  ``X-Idempotency-Key``.
- ``GET /v1/process-reward-models/{model_id}/evals`` — full eval card.

Auth: ``X-Vlabs-Key`` (data plane, mirrors Phases 22 + 29).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.db import (
    ProcessRewardModel,
    ProcessRewardModelRun,
    RewardModel,
    get_db,
)
from vlabs_api.errors import ProcessRewardModelNotFound
from vlabs_api.ids import encode_process_reward_run_id
from vlabs_api.process_reward_inference import (
    ServeOutcome,
    cache_enabled,
    serve_score,
)
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import (
    ProcessRewardEvalsResponse,
    ProcessRewardModelEvalSummary,
    ProcessRewardModelInfo,
    ProcessRewardModelList,
    ProcessRewardModelStatus,
    ProcessRewardModelSummary,
    ProcessRewardScoreBatchRequest,
    ProcessRewardScoreBatchResponse,
    ProcessRewardScoreRequest,
    ProcessRewardScoreResponse,
)

router = APIRouter(tags=["process-reward-models"])

DEFAULT_LIMIT: int = 25
MAX_LIMIT: int = 200
CUSTOMER_VISIBLE_STATUSES: tuple[str, ...] = ("available", "deprecated")


# ── helpers ─────────────────────────────────────────────────────────


def _eval_summary_from_metrics(
    metrics: dict | None,
) -> ProcessRewardModelEvalSummary:
    if not metrics:
        return ProcessRewardModelEvalSummary()
    return ProcessRewardModelEvalSummary(
        processbench_overall=_maybe_float(metrics.get("processbench_overall")),
        bon_lift_vs_phase29=_maybe_float(metrics.get("bon_lift_vs_phase29")),
        aggregate_calibration_coverage=_maybe_float(
            metrics.get("aggregate_calibration_coverage")
        ),
    )


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _resolve_base_rm_model_id(
    session: AsyncSession, base_rm_id
) -> str | None:
    """Look up the Phase 29 RM ``model_id`` for the FK-linked row, so
    PRM detail responses surface the parent RM by its public id."""
    if base_rm_id is None:
        return None
    res = await session.execute(
        select(RewardModel.model_id).where(RewardModel.id == base_rm_id)
    )
    row = res.scalar_one_or_none()
    return row


async def _model_to_info(
    session: AsyncSession, model: ProcessRewardModel
) -> ProcessRewardModelInfo:
    base_rm_model_id = await _resolve_base_rm_model_id(session, model.base_rm_id)
    return ProcessRewardModelInfo(
        model_id=model.model_id,
        name=model.name,
        family=model.family,
        version=model.version,
        base_rm_id=base_rm_model_id,
        step_granularity=model.step_granularity,  # type: ignore[arg-type]
        teacher_source=model.teacher_source,
        student_arch=model.student_arch,
        training_method=model.training_method,
        status=model.status,  # type: ignore[arg-type]
        aggregate_conformal_quantile=model.aggregate_conformal_quantile,
        eval_summary=_eval_summary_from_metrics(model.eval_metrics),
        created_at=model.created_at,
        trained_at=model.trained_at,
        retired_at=model.retired_at,
    )


async def _model_to_summary(
    session: AsyncSession, model: ProcessRewardModel
) -> ProcessRewardModelSummary:
    base_rm_model_id = await _resolve_base_rm_model_id(session, model.base_rm_id)
    return ProcessRewardModelSummary(
        model_id=model.model_id,
        family=model.family,
        version=model.version,
        status=model.status,  # type: ignore[arg-type]
        base_rm_id=base_rm_model_id,
        step_granularity=model.step_granularity,  # type: ignore[arg-type]
        created_at=model.created_at,
        eval_summary=_eval_summary_from_metrics(model.eval_metrics),
    )


async def _resolve_model(
    session: AsyncSession, model_id: str
) -> ProcessRewardModel:
    """Customer-facing lookup: ``training`` and ``retired`` rows
    surface as :class:`ProcessRewardModelNotFound` (we don't leak
    administrative state)."""
    res = await session.execute(
        select(ProcessRewardModel).where(
            ProcessRewardModel.model_id == model_id
        )
    )
    row = res.scalar_one_or_none()
    if row is None or row.status not in CUSTOMER_VISIBLE_STATUSES:
        raise ProcessRewardModelNotFound(detail=f"model_id={model_id!r}")
    return row


def _outcome_to_response(
    outcome: ServeOutcome,
    *,
    audit_uuid_hex: str,
    model_id: str,
) -> ProcessRewardScoreResponse:
    score = outcome.score
    return ProcessRewardScoreResponse(
        step_rewards=list(score.step_rewards),
        step_confidence_intervals=[
            (float(ci[0]), float(ci[1]))
            for ci in score.step_confidence_intervals
        ],
        aggregate_reward=float(score.aggregate_reward),
        aggregate_confidence_interval=(
            float(score.aggregate_confidence_interval[0]),
            float(score.aggregate_confidence_interval[1]),
        ),
        coverage_guarantee=float(score.coverage_guarantee),
        step_count=int(score.step_count),
        model_id=model_id,
        schema_version=score.schema_version,
        cache_hit=outcome.cache_hit,
        latency_ms=outcome.latency_ms,
        audit_id=audit_uuid_hex,
        segmentation_warning=score.segmentation_warning,
    )


# ── GET /v1/process-reward-models ──────────────────────────────────


@router.get("/process-reward-models", response_model=ProcessRewardModelList)
async def list_process_reward_models(
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    family: str | None = Query(default=None, max_length=100),
    status: ProcessRewardModelStatus | None = Query(default=None),
) -> ProcessRewardModelList:
    """Paginated list, sorted ``created_at DESC`` so newest first."""
    base = select(ProcessRewardModel).where(
        ProcessRewardModel.status.in_(CUSTOMER_VISIBLE_STATUSES)
    )
    if family is not None:
        base = base.where(ProcessRewardModel.family == family)
    if status is not None:
        if status not in CUSTOMER_VISIBLE_STATUSES:
            return ProcessRewardModelList(
                items=[], total=0, limit=limit, offset=offset
            )
        base = base.where(ProcessRewardModel.status == status)

    total_query = base.with_only_columns(func.count()).order_by(None)
    total = (await session.execute(total_query)).scalar_one()
    items_query = (
        base.order_by(ProcessRewardModel.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(items_query)).scalars().all()
    summaries = [await _model_to_summary(session, r) for r in rows]
    return ProcessRewardModelList(
        items=summaries,
        total=int(total),
        limit=limit,
        offset=offset,
    )


# ── GET /v1/process-reward-models/{id} ─────────────────────────────


@router.get(
    "/process-reward-models/{model_id}", response_model=ProcessRewardModelInfo
)
async def get_process_reward_model(
    model_id: str,
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001
    session: AsyncSession = Depends(get_db),
) -> ProcessRewardModelInfo:
    """Single-model detail surface — eval summary + lifecycle dates."""
    row = await _resolve_model(session, model_id)
    return await _model_to_info(session, row)


# ── POST /v1/process-reward-models/{id}/score ──────────────────────


@router.post(
    "/process-reward-models/{model_id}/score",
    response_model=ProcessRewardScoreResponse,
    status_code=200,
)
async def score_process_reward_model(
    model_id: str,
    payload: ProcessRewardScoreRequest,
    request: Request,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> ProcessRewardScoreResponse:
    """Score a single (prompt, reasoning_trace) pair. Stub-mode in 30.E."""
    model = await _resolve_model(session, model_id)
    cache_on = cache_enabled(request.headers)
    cache_get, cache_set = _resolve_cache_handles(cache_on)
    outcome = await serve_score(
        model=model,
        prompt=payload.prompt,
        reasoning_trace=payload.reasoning_trace,
        cache_get=cache_get,
        cache_set=cache_set,
        cache_on=cache_on,
    )
    audit_uuid_hex = await _persist_audit_row(
        session=session,
        auth=auth,
        model=model,
        outcome=outcome,
        env_id=payload.env_id,
        idempotency_key=request.headers.get("x-idempotency-key"),
    )
    await session.commit()
    return _outcome_to_response(
        outcome, audit_uuid_hex=audit_uuid_hex, model_id=model.model_id
    )


# ── POST /v1/process-reward-models/{id}/score/batch ────────────────


@router.post(
    "/process-reward-models/{model_id}/score/batch",
    response_model=ProcessRewardScoreBatchResponse,
    status_code=200,
)
async def score_process_reward_model_batch(
    model_id: str,
    payload: ProcessRewardScoreBatchRequest,
    request: Request,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> ProcessRewardScoreBatchResponse:
    """Batch scoring; idempotent on ``X-Idempotency-Key``."""
    model = await _resolve_model(session, model_id)
    cache_on = cache_enabled(request.headers)
    cache_get, cache_set = _resolve_cache_handles(cache_on)

    responses: list[ProcessRewardScoreResponse] = []
    schema_version = "v0.1.0-stub"
    for item in payload.items:
        outcome = await serve_score(
            model=model,
            prompt=item.prompt,
            reasoning_trace=item.reasoning_trace,
            cache_get=cache_get,
            cache_set=cache_set,
            cache_on=cache_on,
        )
        audit_uuid_hex = await _persist_audit_row(
            session=session,
            auth=auth,
            model=model,
            outcome=outcome,
            env_id=item.env_id,
            idempotency_key=request.headers.get("x-idempotency-key"),
        )
        responses.append(
            _outcome_to_response(
                outcome,
                audit_uuid_hex=audit_uuid_hex,
                model_id=model.model_id,
            )
        )
        schema_version = outcome.score.schema_version

    await session.commit()
    return ProcessRewardScoreBatchResponse(
        items=responses,
        total=len(responses),
        model_id=model.model_id,
        schema_version=schema_version,
    )


# ── GET /v1/process-reward-models/{id}/evals ───────────────────────


@router.get(
    "/process-reward-models/{model_id}/evals",
    response_model=ProcessRewardEvalsResponse,
)
async def get_process_reward_model_evals(
    model_id: str,
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001
    session: AsyncSession = Depends(get_db),
) -> ProcessRewardEvalsResponse:
    """Full eval card — per-env breakdown + ProcessBench detail + BoN
    comparisons + per-step + aggregate calibration trace."""
    row = await _resolve_model(session, model_id)
    metrics = row.eval_metrics or {}
    return ProcessRewardEvalsResponse(
        model_id=row.model_id,
        eval_summary=_eval_summary_from_metrics(metrics),
        held_out_envs=metrics.get("held_out_envs", {}) or {},
        processbench=metrics.get("processbench", {}) or {},
        bon=metrics.get("bon", {}) or {},
        calibration=metrics.get("calibration", {}) or {},
    )


# ── helpers (cache + audit) ────────────────────────────────────────


def _resolve_cache_handles(cache_on: bool):
    if not cache_on:
        return None, None
    try:
        from vlabs_api.redis_client import get_client  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None, None
    client = get_client()
    if client is None:
        return None, None

    async def cache_get(key: str):
        try:
            res = await client.pipeline(["GET", key])
        except Exception:  # noqa: BLE001
            return None
        return _extract_pipeline_value(res)

    async def cache_set(key: str, value: str, ttl_seconds: int) -> None:
        try:
            await client.pipeline(["SETEX", key, str(ttl_seconds), value])
        except Exception:  # noqa: BLE001
            return None

    return cache_get, cache_set


def _extract_pipeline_value(payload):
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("result")
    return None


async def _persist_audit_row(
    *,
    session: AsyncSession,
    auth: AuthContext,
    model: ProcessRewardModel,
    outcome: ServeOutcome,
    env_id: str | None,
    idempotency_key: str | None,
) -> str:
    score = outcome.score
    run = ProcessRewardModelRun(
        process_reward_model_id=model.id,
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        prompt_hash=outcome.hashes.prompt_hash,
        trace_hash=outcome.hashes.trace_hash,
        env_id=env_id,
        step_count=int(score.step_count),
        step_rewards=[float(r) for r in score.step_rewards],
        step_cis=[
            [float(ci[0]), float(ci[1])]
            for ci in score.step_confidence_intervals
        ],
        aggregate_reward=float(score.aggregate_reward),
        aggregate_ci_low=float(score.aggregate_confidence_interval[0]),
        aggregate_ci_high=float(score.aggregate_confidence_interval[1]),
        coverage_guarantee=float(score.coverage_guarantee),
        cache_hit=bool(outcome.cache_hit),
        latency_ms=int(outcome.latency_ms),
        idempotency_key=idempotency_key,
    )
    session.add(run)
    await session.flush()
    return encode_process_reward_run_id(run.id)


__all__ = ["router"]
