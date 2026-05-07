"""Orchestration for ``POST /v1/score`` — env load → parse → score → persist.

PHASE_22_PLAN.md §5.2 server-side flow:
  1. Validate env_id in registry; load env factory.
  2. Re-derive instance from seed (stateless).
  3. Run env.adapter.parse_response(completion, instance).
  4. Compute reward via env.score(prediction, instance).
  5. Apply conformal calibration (env.conformal_quantile).
  6. Persist audit_calls row, return audit_id.

Latency: hard 30 s timeout per call. Reward clamped to ``[0, 1]``;
``NaN`` → 0. Per-env semaphore from
:mod:`vlabs_api.concurrency` gates imaging envs to 4 concurrent calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.concurrency import get_semaphore
from vlabs_api.db import AuditCall
from vlabs_api.errors import UnknownEnvironment

# Hard timeout for the entire env.score() pipeline (s).
DEFAULT_SCORE_TIMEOUT_S: float = 30.0

# Idempotency dedup window (PHASE_22_PLAN.md §5.5).
IDEMPOTENCY_WINDOW = timedelta(hours=24)


def hash_completion(completion: str) -> str:
    """SHA-256 hex digest of the completion text.

    The audit_calls row stores ONLY this hash, never the raw text
    (PHASE_22_PLAN.md §5.3 GDPR / completion privacy guarantee).
    """
    return hashlib.sha256(completion.encode("utf-8")).hexdigest()


def _coerce_reward(value: Any) -> float:
    """Clamp reward to ``[0, 1]``; map non-finite to 0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(0.0, min(1.0, f))


def _components_to_jsonable(components: Any) -> dict[str, float]:
    """Convert env-emitted components dict to plain ``{str: float}``."""
    if not isinstance(components, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in components.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _conformal_interval(
    reward: float, conformal_quantile: float
) -> tuple[float, float]:
    """Build a [low, high] interval in [0, 1] around ``reward``.

    For env families whose score is a partial-credit sum on [0, 1]
    (math envs and the inverse-problem family alike), this is the
    cleanest single-scalar coverage report. The width is ``q̂_α``
    (cached per-env at load_environment time).
    """
    q = float(conformal_quantile)
    return (max(0.0, reward - q), min(1.0, reward + q))


def _alpha_from_env(env: Any) -> float:
    """Pull the calibration α from the env's hyperparams (fallback 0.1)."""
    hp = getattr(env, "hyperparams", None)
    if isinstance(hp, dict) and "alpha" in hp:
        try:
            return float(hp["alpha"])
        except (TypeError, ValueError):
            pass
    return 0.1


async def find_idempotent_audit(
    session: AsyncSession,
    user_id: uuid.UUID,
    idempotency_key: str | None,
) -> AuditCall | None:
    """Return any audit row matching ``(idempotency_key, user_id)``.

    The 24 h window check is applied separately by
    :func:`is_within_idempotency_window` — this helper just locates a
    candidate so that out-of-window rows can be explicitly cleared
    before the fresh insert (the partial unique index on
    ``(idempotency_key, user_id)`` blocks two non-null rows from
    coexisting per spec §5.3).
    """
    if not idempotency_key:
        return None
    res = await session.execute(
        select(AuditCall)
        .where(AuditCall.user_id == user_id)
        .where(AuditCall.idempotency_key == idempotency_key)
        .order_by(AuditCall.created_at.desc())
        .limit(1)
    )
    return res.scalar_one_or_none()


def is_within_idempotency_window(row: AuditCall) -> bool:
    """True iff ``row.created_at`` is within :const:`IDEMPOTENCY_WINDOW`."""
    cutoff = datetime.now(UTC) - IDEMPOTENCY_WINDOW
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at >= cutoff


def _safe_get_adapter(env_id: str) -> Any:
    """Trigger adapter registration once and return the adapter (or None)."""
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  registers all
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS, get_adapter

    if env_id not in _ADAPTERS:
        return None
    return get_adapter(env_id)


def _load_env(env_id: str) -> Any:
    """Resolve env_id; lift to UnknownEnvironment on miss."""
    from verifiable_labs_envs import list_environments, load_environment

    if env_id not in list_environments():
        raise UnknownEnvironment(detail=f"env_id={env_id!r}")
    return load_environment(env_id, calibration_quantile=0.5)


def _score_sync(env_id: str, seed: int, completion: str) -> dict[str, Any]:
    """The CPU work: load env, parse, score. Synchronous so it can run
    cleanly in :func:`asyncio.to_thread` — blocking numpy/SymPy work
    does not stall the event loop."""
    env = _load_env(env_id)
    instance = env.generate_instance(seed=seed)

    adapter = _safe_get_adapter(env_id)
    if adapter is None:
        # Defensive: an env without a registered adapter cannot be
        # scored from text. Raise UnknownEnvironment so the caller
        # serves a 404 rather than 5xx — same surface a missing env_id
        # would yield.
        raise UnknownEnvironment(detail=f"no adapter registered for env_id={env_id!r}")

    from verifiable_labs_envs.solvers.llm_solver import LLMSolverError

    try:
        prediction = adapter.parse_response(completion, instance)
        scored = env.score(prediction, instance)
    except LLMSolverError as exc:
        # Adversarial / malformed completion: synthesise a zero score
        # rather than 5xx. Audit row still written so the customer can
        # see the failure mode in their dashboard.
        scored = {
            "reward": 0.0,
            "components": {"format_valid": 0.0, "parse_valid": 0.0, "correct": 0.0},
            "meta": {"parse_error": type(exc).__name__},
        }

    scored["_conformal_quantile"] = float(getattr(env, "conformal_quantile", 0.5))
    return scored


async def score_completion(
    env_id: str,
    seed: int,
    completion: str,
    *,
    timeout_s: float = DEFAULT_SCORE_TIMEOUT_S,
) -> tuple[dict[str, Any], int, float]:
    """Score a completion end-to-end.

    Returns ``(scored_dict, latency_ms, conformal_quantile)``. Wraps the
    env work in :func:`asyncio.to_thread` so blocking numpy/SymPy work
    does not stall the event loop; gates concurrent calls per-env via
    the semaphore from :mod:`vlabs_api.concurrency`.
    """
    sem = get_semaphore(env_id)
    start = time.perf_counter()
    async with sem:
        try:
            scored = await asyncio.wait_for(
                asyncio.to_thread(_score_sync, env_id, seed, completion),
                timeout=timeout_s,
            )
        except TimeoutError:
            scored = {
                "reward": 0.0,
                "components": {"format_valid": 0.0, "parse_valid": 0.0, "correct": 0.0},
                "meta": {"timeout": True, "timeout_s": float(timeout_s)},
                "_conformal_quantile": 0.5,
            }
    latency_ms = int((time.perf_counter() - start) * 1000)
    quantile = float(scored.pop("_conformal_quantile", 0.5))

    return scored, latency_ms, quantile


__all__ = [
    "DEFAULT_SCORE_TIMEOUT_S",
    "IDEMPOTENCY_WINDOW",
    "hash_completion",
    "find_idempotent_audit",
    "is_within_idempotency_window",
    "score_completion",
]


# Public helpers exposed for the route-level handler.
__all__ += ["_coerce_reward", "_components_to_jsonable", "_conformal_interval", "_alpha_from_env"]
