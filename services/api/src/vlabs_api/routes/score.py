"""``POST /v1/score`` — per-call calibrated reward + audit row (Phase 22.C).

Auth: ``X-Vlabs-Key`` (data plane) → ``enforce_rate_limit``. Counts
against the per-tier ``scores_per_month`` quota (shared with
``/v1/instance``). Idempotent re-issues with the same
``X-Idempotency-Key`` within 24 h return the cached audit row WITHOUT
incrementing the counter.

PHASE_22_PLAN.md §5.2 + §5.3 + §5.5 lock the contract; this handler
is the thin wrapper around :mod:`vlabs_api.scoring`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import AuditCall, get_db
from vlabs_api.errors import QuotaExceeded
from vlabs_api.ids import encode_audit_id
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import ScoreRequest, ScoreResponse
from vlabs_api.scoring import (
    _alpha_from_env,
    _coerce_reward,
    _components_to_jsonable,
    _conformal_interval,
    find_idempotent_audit,
    hash_completion,
    is_within_idempotency_window,
    score_completion,
)
from vlabs_api.usage import (
    get_current_counter,
    increment_scores_counter,
    tier_scores_limit,
)

router = APIRouter(tags=["training"])


def _audit_to_response(row: AuditCall, env_version: str) -> ScoreResponse:
    return ScoreResponse(
        reward=float(row.reward),
        conformal_interval=(float(row.conformal_low), float(row.conformal_high)),
        coverage_guarantee=float(row.coverage),
        audit_id=encode_audit_id(row.id),
        components_breakdown={k: float(v) for k, v in row.components_json.items()},
        env_version=env_version,
        latency_ms=int(row.latency_ms),
    )


@router.post("/score", response_model=ScoreResponse)
async def score_endpoint(
    request: Request,  # noqa: ARG001  (kept for parity with rest-API conventions)
    payload: ScoreRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> ScoreResponse:
    from verifiable_labs_envs import __version__ as env_version

    # 1. Idempotency hit — return the cached audit row WITHOUT scoring,
    # WITHOUT incrementing the counter. PHASE_22_PLAN.md §5.5.
    # Out-of-window rows are deleted before the fresh insert so the
    # partial unique index on (idempotency_key, user_id) doesn't block.
    cached = await find_idempotent_audit(session, auth.user_id, payload.idempotency_key)
    if cached is not None:
        if is_within_idempotency_window(cached):
            return _audit_to_response(cached, env_version=cached.env_version)
        # Stale row: clear it to make room for the fresh insert below.
        await session.delete(cached)
        await session.flush()

    # 2. Quota pre-flight (shared scores_per_month with /v1/instance).
    counter = await get_current_counter(session, auth.api_key_id)
    used = counter.scores_count if counter else 0
    cap = tier_scores_limit(auth.tier)
    if used + 1 > cap:
        raise QuotaExceeded(
            detail=(
                f"tier={auth.tier} scores_cap={cap}, used={used}; "
                "upgrade or wait for next month"
            )
        )

    # 3. Score the completion (env load + adapter + reward + timeout).
    scored, latency_ms, conformal_quantile = await score_completion(
        env_id=payload.env_id,
        seed=payload.seed,
        completion=payload.completion,
    )

    # 4. Resolve reward, components, interval.
    reward = _coerce_reward(scored.get("reward", 0.0))
    components = _components_to_jsonable(scored.get("components", {}))
    interval = _conformal_interval(reward, conformal_quantile)

    # alpha from env hyperparams via a lightweight env reload (cached).
    from vlabs_api.scoring import _load_env

    alpha = _alpha_from_env(_load_env(payload.env_id))
    coverage_guarantee = max(0.0, min(1.0, 1.0 - alpha))

    # 5. Persist audit row.
    audit = AuditCall(
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        env_id=payload.env_id,
        env_version=env_version,
        seed=int(payload.seed),
        completion_hash=hash_completion(payload.completion),
        reward=reward,
        conformal_low=float(interval[0]),
        conformal_high=float(interval[1]),
        coverage=coverage_guarantee,
        components_json=components,
        latency_ms=int(latency_ms),
        idempotency_key=payload.idempotency_key,
    )
    session.add(audit)

    # 6. Bump scores counter.
    await increment_scores_counter(session, auth.api_key_id, delta=1)
    await session.commit()
    await session.refresh(audit)

    return ScoreResponse(
        reward=reward,
        conformal_interval=interval,
        coverage_guarantee=coverage_guarantee,
        audit_id=encode_audit_id(audit.id),
        components_breakdown=components,
        env_version=env_version,
        latency_ms=int(latency_ms),
    )
