"""``POST /v1/instance`` — procedural problem-instance fetch (Phase 22.B).

Stateless: re-derives the instance deterministically from
``(env_id, seed, difficulty_kwargs)`` via
``verifiable_labs_envs.load_environment(env_id).generate_instance(seed)``.
No server-side cache.

Auth: ``X-Vlabs-Key`` (data plane) → ``enforce_rate_limit``. Counts
against the per-tier ``scores_per_month`` quota (shared with
``/v1/score``).

Latency target: <50 ms p95 for symbolic envs; <300 ms for imaging
envs (driven by env baseline / data load).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import get_db
from vlabs_api.errors import QuotaExceeded, UnknownEnvironment
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import InstanceRequest, InstanceResponse
from vlabs_api.usage import (
    get_current_counter,
    increment_scores_counter,
    tier_scores_limit,
)

router = APIRouter(tags=["training"])


def _load_env_for_instance(env_id: str) -> Any:
    """Resolve env_id through the in-tree registry; raise on miss.

    ``difficulty_kwargs`` are passed to ``env.generate_instance``, NOT
    here — most envs' ``load_environment`` factories accept only
    ``calibration_quantile`` + ``fast``, and we'd otherwise reject
    perfectly valid per-instance kwargs.
    """
    # Lazy import keeps the FastAPI app cold-start fast and avoids
    # pulling numpy/sympy at module load on routes that never run.
    from verifiable_labs_envs import list_environments, load_environment

    if env_id not in list_environments():
        raise UnknownEnvironment(detail=f"env_id={env_id!r}")

    # Fast path: skip the env's own conformal calibration; instance
    # generation does not depend on quantile, and calibration warm-up
    # adds ~30 baseline rollouts of latency that we don't need here.
    return load_environment(env_id, calibration_quantile=0.5)


def _public_metadata(instance: Any) -> dict[str, Any]:
    """Return the env's ``Instance.as_inputs()`` with oracle fields stripped.

    Falls back to ``{}`` if the instance has no ``as_inputs`` method
    (defensive — every shipped env defines one). Numpy arrays are
    coerced to lists so the JSON serialiser is happy.
    """
    if not hasattr(instance, "as_inputs"):
        return {}
    raw = instance.as_inputs()
    return _coerce_to_jsonable(raw)


def _coerce_to_jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy arrays / scalars into JSON-safe types."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover — numpy is a hard dep
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _coerce_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_jsonable(v) for v in value]
    return value


@router.post("/instance", response_model=InstanceResponse)
async def instance_endpoint(
    payload: InstanceRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> InstanceResponse:
    # 1. Quota pre-flight (shared scores_per_month with /v1/score).
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

    # 2. Resolve env + render instance. difficulty_kwargs go to
    # generate_instance, which is where per-instance hyperparams live.
    env = _load_env_for_instance(payload.env_id)
    try:
        instance = env.generate_instance(
            seed=payload.seed, **payload.difficulty_kwargs
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise UnknownEnvironment(
            detail=f"env_id={payload.env_id!r} rejected difficulty_kwargs: {exc}"
        ) from exc

    # 3. Render LLM prompt via the registered adapter (lazy import).
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  triggers registration
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS, get_adapter

    if payload.env_id not in _ADAPTERS:
        # No adapter registered: surface the env's prompt field directly
        # (math envs all have one; imaging envs always have an adapter).
        prompt = str(getattr(instance, "prompt", ""))
    else:
        adapter = get_adapter(payload.env_id)
        prompt = adapter.build_user_prompt(instance)

    # 4. Bump the scores counter — same UPSERT pattern as /v1/calibrate.
    await increment_scores_counter(session, auth.api_key_id, delta=1)
    await session.commit()

    # 5. Compose response.
    from verifiable_labs_envs import __version__ as env_version

    return InstanceResponse(
        instance_seed=int(payload.seed),
        prompt=prompt,
        metadata=_public_metadata(instance),
        env_version=env_version,
    )
