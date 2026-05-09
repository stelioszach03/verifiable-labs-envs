"""``/v1/monitors`` — continuous capability monitoring (Phase 28).

- **28.B (this file)** ships ``POST /v1/monitors`` (create),
  ``GET /v1/monitors`` (paginated list), ``GET /v1/monitors/{id}``
  (single detail), ``PATCH /v1/monitors/{id}`` (partial update / pause
  / rotate token / rebaseline), ``DELETE /v1/monitors/{id}`` (soft
  delete).
- **28.C** adds ``POST /v1/monitors/{id}/run`` (trigger ad-hoc audit)
  + the internal scheduler webhook.
- **28.E** adds ``GET /v1/monitors/{id}/runs`` (paginated history)
  + ``GET /v1/monitors/{id}/runs/{rid}`` (single run detail).

PHASE_28_PLAN.md §6 schema, §8 endpoint contracts, §5 D8-C tier-cap
enforcement at create-time. The customer auth token is encrypted at
rest via the existing Fernet helper from Phase 23
(``vlabs_api.llm_key_crypto``); the response surface returns only the
``auth_token_fingerprint`` (first 8 hex chars of SHA-256(token)) so
the customer can confirm "is this the key I think it is?" without
ever leaking the secret.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.auth import AuthContext
from vlabs_api.db import Monitor, get_db
from vlabs_api.errors import (
    MonitorInvalidState,
    MonitorNameConflict,
    MonitorNotFound,
    MonitorTierExceeded,
    UnknownEnvironment,
)
from vlabs_api.ids import (
    encode_monitor_id,
    encode_monitor_run_id,
    parse_monitor_id,
)
from vlabs_api.llm_key_crypto import encrypt_llm_api_key
from vlabs_api.monitor_cadence import (
    compute_next_run_at,
    projected_monthly_episodes,
)
from vlabs_api.ratelimit import enforce_rate_limit
from vlabs_api.schemas import (
    MonitorAlertChannel,
    MonitorAlertChannelInfo,
    MonitorCreateRequest,
    MonitorCreateResponse,
    MonitorList,
    MonitorResponse,
    MonitorSummary,
    MonitorUpdateRequest,
)
from vlabs_api.usage import tier_monitor_caps

router = APIRouter(tags=["monitors"])

DEFAULT_LIMIT = 25
MAX_LIMIT = 100


# ── helpers ─────────────────────────────────────────────────────────


def _validate_env_subset(env_subset: list[str]) -> None:
    from verifiable_labs_envs import list_environments

    registered = set(list_environments())
    for env_id in env_subset:
        if env_id not in registered:
            raise UnknownEnvironment(detail=f"env_id={env_id!r}")


def _fingerprint(token: str) -> str:
    """First 8 hex chars of SHA-256 — used as opaque key identifier."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def _project_episodes(monitor: Monitor) -> int:
    return projected_monthly_episodes(
        monitor.cadence,  # type: ignore[arg-type]
        len(monitor.env_subset),
        int(monitor.episodes_per_env),
    )


def _enforce_tier_caps(
    *,
    cadence: str,
    env_subset: list[str],
    episodes_per_env: int,
    active_count: int,
    tier: str,
) -> tuple[int, int]:
    """Apply PHASE_28_PLAN.md §5 D8-C / §12 tier ceilings.

    Returns ``(projected_monthly_episodes, monthly_episode_cap)``.
    Raises :class:`MonitorTierExceeded` if any of the four guards
    (count cap, env subset size, episodes per env, projected monthly
    episode cap) trips.
    """
    monitors_max, envs_max, episodes_max = tier_monitor_caps(tier)  # type: ignore[arg-type]
    if active_count >= monitors_max:
        raise MonitorTierExceeded(
            detail=(
                f"tier={tier} monitors_max={monitors_max}; you already "
                f"have {active_count} active monitor(s). Upgrade your "
                "tier or pause an existing monitor first."
            )
        )
    if len(env_subset) > envs_max:
        raise MonitorTierExceeded(
            detail=(
                f"tier={tier} monitor_envs_max={envs_max}; requested "
                f"{len(env_subset)} envs."
            )
        )
    if episodes_per_env > episodes_max:
        raise MonitorTierExceeded(
            detail=(
                f"tier={tier} monitor_episodes_max={episodes_max}; "
                f"requested {episodes_per_env}."
            )
        )

    projected = projected_monthly_episodes(
        cadence,  # type: ignore[arg-type]
        len(env_subset),
        episodes_per_env,
    )
    monthly_cap = monitors_max * envs_max * episodes_max * 30
    if projected > monthly_cap:
        raise MonitorTierExceeded(
            detail=(
                f"projected {projected} episodes/month exceeds tier "
                f"cap of {monthly_cap}. Lower cadence, env_subset, or "
                "episodes_per_env."
            )
        )
    return projected, monthly_cap


def _alert_channels_to_storage(
    channels: list[MonitorAlertChannel],
) -> list[dict[str, Any]]:
    """Persist-shape transformation.

    The Slack ``webhook_url`` is encrypted at rest at the row layer in
    28.D; in 28.B we store the raw URL but always strip it from any
    response surface (only ``webhook_url_fingerprint`` is exposed).
    """
    out: list[dict[str, Any]] = []
    for ch in channels:
        entry: dict[str, Any] = {"type": ch.type}
        if ch.type == "email":
            if not ch.address:
                raise MonitorInvalidState(
                    detail="alert_channels[type=email] requires 'address'"
                )
            entry["address"] = ch.address
        elif ch.type == "slack":
            if not ch.webhook_url:
                raise MonitorInvalidState(
                    detail="alert_channels[type=slack] requires 'webhook_url'"
                )
            entry["webhook_url"] = ch.webhook_url
            entry["webhook_url_fingerprint"] = _fingerprint(ch.webhook_url)
        out.append(entry)
    return out


def _alert_channels_to_response(
    stored: list[dict[str, Any]] | None,
) -> list[MonitorAlertChannelInfo]:
    if not stored:
        return []
    out: list[MonitorAlertChannelInfo] = []
    for entry in stored:
        out.append(
            MonitorAlertChannelInfo(
                type=entry["type"],
                address=entry.get("address"),
                webhook_url_fingerprint=entry.get("webhook_url_fingerprint"),
            )
        )
    return out


def _monitor_to_response(monitor: Monitor) -> MonitorResponse:
    return MonitorResponse(
        monitor_id=encode_monitor_id(monitor.id),
        name=monitor.name,
        model_endpoint=monitor.model_endpoint,
        model_name=monitor.model_name,
        auth_token_fingerprint=monitor.auth_token_fingerprint,
        cadence=monitor.cadence,  # type: ignore[arg-type]
        env_subset=list(monitor.env_subset),
        episodes_per_env=int(monitor.episodes_per_env),
        alert_channels=_alert_channels_to_response(monitor.alert_channels),
        status=monitor.status,  # type: ignore[arg-type]
        retention_days=int(monitor.retention_days),
        baseline_run_id=(
            encode_monitor_run_id(monitor.baseline_run_id)
            if monitor.baseline_run_id is not None
            else None
        ),
        created_at=monitor.created_at,
        updated_at=monitor.updated_at,
        last_run_at=monitor.last_run_at,
        next_run_at=monitor.next_run_at,
        projected_monthly_episodes=_project_episodes(monitor),
    )


def _monitor_to_summary(monitor: Monitor) -> MonitorSummary:
    return MonitorSummary(
        monitor_id=encode_monitor_id(monitor.id),
        name=monitor.name,
        model_name=monitor.model_name,
        cadence=monitor.cadence,  # type: ignore[arg-type]
        status=monitor.status,  # type: ignore[arg-type]
        env_subset=list(monitor.env_subset),
        episodes_per_env=int(monitor.episodes_per_env),
        last_run_at=monitor.last_run_at,
        next_run_at=monitor.next_run_at,
        created_at=monitor.created_at,
    )


async def _count_active_monitors(
    session: AsyncSession, user_id: Any,
) -> int:
    res = await session.execute(
        select(func.count(Monitor.id))
        .where(Monitor.user_id == user_id)
        .where(Monitor.status == "active")
    )
    return int(res.scalar_one())


async def _load_monitor(
    session: AsyncSession,
    monitor_id_str: str,
    user_id: Any,
) -> Monitor:
    monitor_uuid = parse_monitor_id(monitor_id_str)
    res = await session.execute(
        select(Monitor)
        .where(Monitor.id == monitor_uuid)
        .where(Monitor.user_id == user_id)
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise MonitorNotFound(detail=f"monitor_id={monitor_id_str}")
    return row


# ── routes ─────────────────────────────────────────────────────────


@router.post(
    "/monitors",
    response_model=MonitorCreateResponse,
    status_code=201,
)
async def create_monitor(
    payload: MonitorCreateRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> MonitorCreateResponse:
    # 1. Validate env subset against the registry.
    _validate_env_subset(payload.env_subset)

    # 2. Tier-cap pre-flight.
    active_count = await _count_active_monitors(session, auth.user_id)
    projected, _cap = _enforce_tier_caps(
        cadence=payload.cadence,
        env_subset=payload.env_subset,
        episodes_per_env=payload.episodes_per_env,
        active_count=active_count,
        tier=auth.tier,
    )

    # 3. Duplicate-name guard (per-user).
    name_clash = await session.execute(
        select(Monitor.id)
        .where(Monitor.user_id == auth.user_id)
        .where(Monitor.name == payload.name)
    )
    if name_clash.scalar_one_or_none() is not None:
        raise MonitorNameConflict(detail=f"name={payload.name!r}")

    # 4. Encrypt the auth token + persist.
    now = datetime.now(UTC)
    monitor = Monitor(
        user_id=auth.user_id,
        api_key_id=auth.api_key_id,
        name=payload.name,
        model_endpoint=payload.model_endpoint,
        model_name=payload.model_name,
        auth_token_encrypted=encrypt_llm_api_key(payload.auth_token),
        auth_token_fingerprint=_fingerprint(payload.auth_token),
        cadence=payload.cadence,
        env_subset=list(payload.env_subset),
        episodes_per_env=int(payload.episodes_per_env),
        alert_channels=_alert_channels_to_storage(payload.alert_channels),
        status="active",
        next_run_at=compute_next_run_at(payload.cadence, anchor=now),
    )
    session.add(monitor)
    await session.commit()
    await session.refresh(monitor)

    _, _, episodes_max = tier_monitor_caps(auth.tier)  # type: ignore[arg-type]
    monitors_max, envs_max, _ = tier_monitor_caps(auth.tier)  # type: ignore[arg-type]
    return MonitorCreateResponse(
        monitor_id=encode_monitor_id(monitor.id),
        name=monitor.name,
        status=monitor.status,  # type: ignore[arg-type]
        cadence=monitor.cadence,  # type: ignore[arg-type]
        next_run_at=monitor.next_run_at,
        auth_token_fingerprint=monitor.auth_token_fingerprint,
        projected_monthly_episodes=projected,
        tier_limit_episodes=monitors_max * envs_max * episodes_max * 30,
        created_at=monitor.created_at,
    )


@router.get("/monitors", response_model=MonitorList)
async def list_monitors(
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None, max_length=32),
) -> MonitorList:
    """Paginated list, sorted ``created_at DESC`` (matches /v1/datasets)."""
    base = select(Monitor).where(Monitor.user_id == auth.user_id)
    count_base = select(func.count(Monitor.id)).where(
        Monitor.user_id == auth.user_id
    )
    if status is not None:
        base = base.where(Monitor.status == status)
        count_base = count_base.where(Monitor.status == status)
    total = int((await session.execute(count_base)).scalar_one())
    res = await session.execute(
        base.order_by(Monitor.created_at.desc()).limit(limit).offset(offset)
    )
    rows = res.scalars().all()
    return MonitorList(
        items=[_monitor_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/monitors/{monitor_id}", response_model=MonitorResponse)
async def get_monitor(
    monitor_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> MonitorResponse:
    monitor = await _load_monitor(session, monitor_id, auth.user_id)
    return _monitor_to_response(monitor)


@router.patch("/monitors/{monitor_id}", response_model=MonitorResponse)
async def update_monitor(
    monitor_id: str,
    payload: MonitorUpdateRequest,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> MonitorResponse:
    monitor = await _load_monitor(session, monitor_id, auth.user_id)

    # Validate env subset / episode count against tier caps when either
    # is changing. We use the post-update values for the projection.
    next_cadence = payload.cadence or monitor.cadence
    next_envs = (
        list(payload.env_subset)
        if payload.env_subset is not None
        else list(monitor.env_subset)
    )
    next_episodes = (
        int(payload.episodes_per_env)
        if payload.episodes_per_env is not None
        else int(monitor.episodes_per_env)
    )

    if payload.env_subset is not None:
        _validate_env_subset(payload.env_subset)

    # Tier caps revalidated whenever any cap-relevant field changes.
    if (
        payload.cadence is not None
        or payload.env_subset is not None
        or payload.episodes_per_env is not None
    ):
        active_count = await _count_active_monitors(session, auth.user_id)
        # The monitor we're updating already counts as active — exclude it
        # from the count so tier-cap math stays correct on edit.
        if monitor.status == "active":
            active_count -= 1
        _enforce_tier_caps(
            cadence=next_cadence,  # type: ignore[arg-type]
            env_subset=next_envs,
            episodes_per_env=next_episodes,
            active_count=active_count,
            tier=auth.tier,
        )

    if payload.name is not None and payload.name != monitor.name:
        clash = await session.execute(
            select(Monitor.id)
            .where(Monitor.user_id == auth.user_id)
            .where(Monitor.name == payload.name)
            .where(Monitor.id != monitor.id)
        )
        if clash.scalar_one_or_none() is not None:
            raise MonitorNameConflict(detail=f"name={payload.name!r}")
        monitor.name = payload.name

    if payload.cadence is not None and payload.cadence != monitor.cadence:
        monitor.cadence = payload.cadence
        # Recompute next_run_at against the new cadence.
        monitor.next_run_at = compute_next_run_at(
            payload.cadence, anchor=datetime.now(UTC),
        )

    if payload.env_subset is not None:
        monitor.env_subset = list(payload.env_subset)

    if payload.episodes_per_env is not None:
        monitor.episodes_per_env = int(payload.episodes_per_env)

    if payload.alert_channels is not None:
        monitor.alert_channels = _alert_channels_to_storage(
            payload.alert_channels
        )

    if payload.auth_token is not None:
        monitor.auth_token_encrypted = encrypt_llm_api_key(payload.auth_token)
        monitor.auth_token_fingerprint = _fingerprint(payload.auth_token)

    if payload.status is not None and payload.status != monitor.status:
        if payload.status not in ("active", "paused"):
            raise MonitorInvalidState(
                detail=(
                    f"status={payload.status!r}: only 'active' or "
                    "'paused' may be set via PATCH"
                )
            )
        monitor.status = payload.status
        if payload.status == "active":
            # Reactivating: kick the next-run forward from now.
            monitor.next_run_at = compute_next_run_at(
                monitor.cadence,  # type: ignore[arg-type]
                anchor=datetime.now(UTC),
            )

    if payload.rebaseline:
        monitor.baseline_run_id = None

    monitor.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(monitor)
    return _monitor_to_response(monitor)


@router.delete("/monitors/{monitor_id}", status_code=204)
async def delete_monitor(
    monitor_id: str,
    auth: AuthContext = Depends(enforce_rate_limit),
    session: AsyncSession = Depends(get_db),
) -> None:
    """Soft-delete: status -> 'failed', next_run_at frozen at now+30d.

    Hard-delete is admin-only (out of scope for 28.B). We use 'failed'
    rather than introducing a 'deleted' status so the existing
    monitors_status_check constraint stays unchanged.
    """
    monitor = await _load_monitor(session, monitor_id, auth.user_id)
    monitor.status = "failed"
    # Push next_run_at far enough that the scheduler tick won't pick it.
    monitor.next_run_at = datetime.now(UTC).replace(
        year=datetime.now(UTC).year + 1,
    )
    monitor.updated_at = datetime.now(UTC)
    await session.commit()


# Aliased helpers re-exported so 28.C / 28.D / 28.E modules can reuse
# the loader + response shaper without depending on internal closures.
__all__ = [
    "router",
    "_load_monitor",
    "_monitor_to_response",
    "_monitor_to_summary",
    "_alert_channels_to_storage",
    "_alert_channels_to_response",
    "_fingerprint",
    "_enforce_tier_caps",
]
