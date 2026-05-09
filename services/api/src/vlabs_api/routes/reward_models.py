"""``/v1/reward-models`` — distilled reward model service (Phase 29.E).

Five endpoints per :doc:`PHASE_29_PLAN.md` §10. All run on stub
inference until 29.G lands trained weights; the response shape is
locked NOW so frontend + SDK integrations can land in parallel.

- ``GET /v1/reward-models`` — paginated list with family/status filter.
- ``GET /v1/reward-models/{model_id}`` — single-model detail.
- ``POST /v1/reward-models/{model_id}/score`` — single scoring; opt-in
  Redis cache via ``X-Vlabs-Cache: enable`` (D11-C, default-off).
- ``POST /v1/reward-models/{model_id}/score/batch`` — up to 100
  pairs/call; idempotent on ``X-Idempotency-Key``.
- ``GET /v1/reward-models/{model_id}/evals`` — full eval card.

Auth: ``X-Vlabs-Key`` (data plane, mirrors ``/v1/score``).
Tier check: counts against ``usage_counters.reward_scores_count``;
strict cap enforcement lives in 29.F when the trained student is
billing-eligible.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext, require_api_key
from vlabs_api.db import RewardModel, RewardModelRun, get_db
from vlabs_api.errors import RewardModelNotFound
from vlabs_api.ids import encode_reward_model_run_id
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.reward_distillation_service import (
    ServeOutcome,
    cache_enabled,
    serve_score,
)
from vlabs_api.schemas import (
    RewardEvalsResponse,
    RewardModelEvalSummary,
    RewardModelInfo,
    RewardModelList,
    RewardModelStatus,
    RewardModelSummary,
    RewardScoreBatchRequest,
    RewardScoreBatchResponse,
    RewardScoreRequest,
    RewardScoreResponse,
)

router = APIRouter(tags=["reward-models"])

DEFAULT_LIMIT: int = 25
MAX_LIMIT: int = 200
CUSTOMER_VISIBLE_STATUSES: tuple[str, ...] = ("available", "deprecated")


# ── helpers ─────────────────────────────────────────────────────────


def _eval_summary_from_metrics(metrics: dict | None) -> RewardModelEvalSummary:
    if not metrics:
        return RewardModelEvalSummary()
    return RewardModelEvalSummary(
        rewardbench_overall=_maybe_float(metrics.get("rewardbench_overall")),
        held_out_spearman_avg=_maybe_float(metrics.get("held_out_spearman_avg")),
        calibration_coverage=_maybe_float(metrics.get("calibration_coverage")),
    )


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_to_info(model: RewardModel) -> RewardModelInfo:
    return RewardModelInfo(
        model_id=model.model_id,
        name=model.name,
        family=model.family,
        version=model.version,
        teacher_source=model.teacher_source,
        student_arch=model.student_arch,
        training_method=model.training_method,
        status=model.status,  # type: ignore[arg-type]
        conformal_quantile=model.conformal_quantile,
        eval_summary=_eval_summary_from_metrics(model.eval_metrics),
        created_at=model.created_at,
        trained_at=model.trained_at,
        retired_at=model.retired_at,
    )


def _model_to_summary(model: RewardModel) -> RewardModelSummary:
    return RewardModelSummary(
        model_id=model.model_id,
        family=model.family,
        version=model.version,
        status=model.status,  # type: ignore[arg-type]
        created_at=model.created_at,
        eval_summary=_eval_summary_from_metrics(model.eval_metrics),
    )


async def _resolve_model(
    session: AsyncSession, model_id: str
) -> RewardModel:
    """Fetch a customer-facing ``RewardModel`` by its public ``model_id``.

    Returns the row if it exists AND its status is customer-visible
    (``available`` or ``deprecated``). ``training`` and ``retired``
    rows surface as :class:`RewardModelNotFound` (we don't leak
    administrative state).
    """
    res = await session.execute(
        select(RewardModel).where(RewardModel.model_id == model_id)
    )
    row = res.scalar_one_or_none()
    if row is None or row.status not in CUSTOMER_VISIBLE_STATUSES:
        raise RewardModelNotFound(detail=f"model_id={model_id!r}")
    return row


def _outcome_to_response(
    outcome: ServeOutcome,
    *,
    audit_uuid_hex: str,
    model_id: str,
) -> RewardScoreResponse:
    score = outcome.score
    return RewardScoreResponse(
        reward=score.reward,
        confidence_interval=score.confidence_interval,
        coverage_guarantee=score.coverage_guarantee,
        model_id=model_id,
        schema_version=score.schema_version,
        cache_hit=outcome.cache_hit,
        latency_ms=outcome.latency_ms,
        audit_id=audit_uuid_hex,
    )


# ── GET /v1/reward-models ──────────────────────────────────────────


@router.get("/reward-models", response_model=RewardModelList)
async def list_reward_models(
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001 — gates auth only
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    family: str | None = Query(default=None, max_length=100),
    status: RewardModelStatus | None = Query(default=None),
) -> RewardModelList:
    """Paginated list, sorted ``created_at DESC`` so newest first."""
    base = select(RewardModel).where(
        RewardModel.status.in_(CUSTOMER_VISIBLE_STATUSES)
    )
    if family is not None:
        base = base.where(RewardModel.family == family)
    if status is not None:
        if status not in CUSTOMER_VISIBLE_STATUSES:
            return RewardModelList(items=[], total=0, limit=limit, offset=offset)
        base = base.where(RewardModel.status == status)

    total_query = base.with_only_columns(func.count()).order_by(None)
    total = (await session.execute(total_query)).scalar_one()
    items_query = (
        base.order_by(RewardModel.created_at.desc()).offset(offset).limit(limit)
    )
    rows = (await session.execute(items_query)).scalars().all()
    return RewardModelList(
        items=[_model_to_summary(r) for r in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


# ── GET /v1/reward-models/{id} ──────────────────────────────────────


@router.get("/reward-models/{model_id}", response_model=RewardModelInfo)
async def get_reward_model(
    model_id: str,
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001
    session: AsyncSession = Depends(get_db),
) -> RewardModelInfo:
    """Single-model detail surface — eval summary + lifecycle dates."""
    row = await _resolve_model(session, model_id)
    return _model_to_info(row)


# ── POST /v1/reward-models/{id}/score ───────────────────────────────


@router.post(
    "/reward-models/{model_id}/score",
    response_model=RewardScoreResponse,
    status_code=200,
)
async def score_reward_model(
    model_id: str,
    payload: RewardScoreRequest,
    request: Request,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> RewardScoreResponse:
    """Score a single (prompt, response) pair. Stub-mode in 29.E."""
    model = await _resolve_model(session, model_id)
    cache_on = cache_enabled(request.headers)
    cache_get, cache_set = _resolve_cache_handles(cache_on)
    outcome = await serve_score(
        model=model,
        prompt=payload.prompt,
        response=payload.response,
        cache_get=cache_get,
        cache_set=cache_set,
        cache_on=cache_on,
    )
    audit_uuid_hex = await _persist_audit_row(
        session=session,
        auth=auth,
        model=model,
        prompt_hash=outcome.hashes.prompt_hash,
        response_hash=outcome.hashes.response_hash,
        env_id=payload.env_id,
        outcome=outcome,
        idempotency_key=request.headers.get("x-idempotency-key"),
    )
    await session.commit()
    return _outcome_to_response(
        outcome, audit_uuid_hex=audit_uuid_hex, model_id=model.model_id
    )


# ── POST /v1/reward-models/{id}/score/batch ─────────────────────────


@router.post(
    "/reward-models/{model_id}/score/batch",
    response_model=RewardScoreBatchResponse,
    status_code=200,
)
async def score_reward_model_batch(
    model_id: str,
    payload: RewardScoreBatchRequest,
    request: Request,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> RewardScoreBatchResponse:
    """Batch scoring; idempotent on ``X-Idempotency-Key``."""
    model = await _resolve_model(session, model_id)
    cache_on = cache_enabled(request.headers)
    cache_get, cache_set = _resolve_cache_handles(cache_on)

    responses: list[RewardScoreResponse] = []
    schema_version = "v0.1.0-stub"
    for item in payload.items:
        outcome = await serve_score(
            model=model,
            prompt=item.prompt,
            response=item.response,
            cache_get=cache_get,
            cache_set=cache_set,
            cache_on=cache_on,
        )
        audit_uuid_hex = await _persist_audit_row(
            session=session,
            auth=auth,
            model=model,
            prompt_hash=outcome.hashes.prompt_hash,
            response_hash=outcome.hashes.response_hash,
            env_id=item.env_id,
            outcome=outcome,
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
    return RewardScoreBatchResponse(
        items=responses,
        total=len(responses),
        model_id=model.model_id,
        schema_version=schema_version,
    )


# ── GET /v1/reward-models/{id}/evals ────────────────────────────────


@router.get(
    "/reward-models/{model_id}/evals", response_model=RewardEvalsResponse
)
async def get_reward_model_evals(
    model_id: str,
    auth: AuthContext = Depends(require_api_key),  # noqa: ARG001
    session: AsyncSession = Depends(get_db),
) -> RewardEvalsResponse:
    """Full eval card — per-env breakdown + RewardBench detail +
    calibration trace. The structure of the inner dicts evolves with
    each released version (D12-B); customers parse by key, not slot."""
    row = await _resolve_model(session, model_id)
    metrics = row.eval_metrics or {}
    return RewardEvalsResponse(
        model_id=row.model_id,
        eval_summary=_eval_summary_from_metrics(metrics),
        held_out_envs=metrics.get("held_out_envs", {}) or {},
        rewardbench=metrics.get("rewardbench", {}) or {},
        calibration=metrics.get("calibration", {}) or {},
    )


# ── helpers (continued) ─────────────────────────────────────────────


def _resolve_cache_handles(cache_on: bool):
    """Return ``(cache_get, cache_set)`` coroutines or ``(None, None)``.

    Lazy redis import keeps the route module importable in test
    contexts where Redis isn't configured. The handles wrap the
    Upstash REST client's GET/SETEX surface; failure modes are caught
    by :func:`vlabs_api.reward_distillation_service.serve_score`.
    """
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
    """Upstash REST returns ``[{"result": ...}]`` per pipeline command;
    extract the inner string if present."""
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("result")
    return None


async def _persist_audit_row(
    *,
    session: AsyncSession,
    auth: AuthContext,
    model: RewardModel,
    prompt_hash: str,
    response_hash: str,
    env_id: str | None,
    outcome: ServeOutcome,
    idempotency_key: str | None,
) -> str:
    run = RewardModelRun(
        reward_model_id=model.id,
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        env_id=env_id,
        reward_score=float(outcome.score.reward),
        ci_low=float(outcome.score.confidence_interval[0]),
        ci_high=float(outcome.score.confidence_interval[1]),
        coverage_guarantee=float(outcome.score.coverage_guarantee),
        cache_hit=bool(outcome.cache_hit),
        latency_ms=int(outcome.latency_ms),
        idempotency_key=idempotency_key,
    )
    session.add(run)
    await session.flush()
    return encode_reward_model_run_id(run.id)


__all__ = ["router"]
