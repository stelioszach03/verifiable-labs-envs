"""SQLAlchemy 2.x async engine, session factory, and ORM models.

Models map directly to the schema in
:doc:`PHASE_16_PLAN.md <PHASE_16_PLAN>` §4. Every primary key is a
random UUID generated server-side; ULIDs (sort-friendly) are layered
on top via :mod:`vlabs_api.routes` for public IDs only.

The engine is initialised at FastAPI ``lifespan`` startup via
:func:`init_engine`. Tests override the engine via
:func:`override_engine` against a ``pgserver`` instance.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now_utc() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(Text)
    clerk_user_id: Mapped[str | None] = mapped_column(Text, unique=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    api_keys: Mapped[list[APIKey]] = relationship(back_populates="user")
    subscriptions: Mapped[list[Subscription]] = relationship(back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="api_keys")
    runs: Mapped[list[CalibrationRun]] = relationship(back_populates="api_key")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="api_key")

    __table_args__ = (Index("api_keys_user_idx", "user_id"),)


class CalibrationRun(Base):
    __tablename__ = "calibration_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    alpha: Mapped[float] = mapped_column(Float, nullable=False)
    nonconformity: Mapped[str] = mapped_column(Text, nullable=False)
    n_calibration: Mapped[int] = mapped_column(Integer, nullable=False)
    quantile: Mapped[float] = mapped_column(Float, nullable=False)
    nonconformity_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    request_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_traces: Mapped[int] = mapped_column(Integer, nullable=False)

    api_key: Mapped[APIKey] = relationship(back_populates="runs")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="calibration")

    __table_args__ = (
        Index("calibration_runs_owner_idx", "api_key_id", "created_at"),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    calibration_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("calibration_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    n: Mapped[int] = mapped_column(Integer, nullable=False)
    empirical_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    target_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    passes: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tolerance: Mapped[float] = mapped_column(Float, nullable=False)
    interval_width_mean: Mapped[float] = mapped_column(Float, nullable=False)
    nonconformity_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    request_traces: Mapped[int] = mapped_column(Integer, nullable=False)

    calibration: Mapped[CalibrationRun] = relationship(back_populates="evaluations")
    api_key: Mapped[APIKey] = relationship(back_populates="evaluations")

    __table_args__ = (
        Index("evaluations_calib_idx", "calibration_id", "created_at"),
    )


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        primary_key=True,
    )
    month: Mapped[date] = mapped_column(Date, primary_key=True)
    traces_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    calibrations_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    evaluations_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    predictions_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Phase 22.B — counts /v1/instance + /v1/score against the per-tier
    # monthly cap. Idempotent re-issues of /v1/score do NOT increment.
    scores_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Phase 23.B — counts successfully generated dataset tuples against
    # the per-tier monthly cap. Failed tuples (LLM timeout, parse error,
    # env scoring failure) do NOT increment.
    tuples_generated: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class AuditCall(Base):
    """Per-call audit row written by ``POST /v1/score`` (Phase 22.C).

    The completion text itself is **never** persisted — only its
    SHA-256 hash. Customers can verify a row matches their completion
    by re-hashing locally; nobody else can recover the text.
    """

    __tablename__ = "audit_calls"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    env_id: Mapped[str] = mapped_column(Text, nullable=False)
    env_version: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    completion_hash: Mapped[str] = mapped_column(Text, nullable=False)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    conformal_low: Mapped[float] = mapped_column(Float, nullable=False)
    conformal_high: Mapped[float] = mapped_column(Float, nullable=False)
    coverage: Mapped[float] = mapped_column(Float, nullable=False)
    components_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        Index("audit_calls_user_idx", "user_id", "created_at"),
        Index("audit_calls_env_idx", "env_id", "created_at"),
        Index(
            "audit_calls_idempotency_idx",
            "idempotency_key",
            "user_id",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class DatasetJob(Base):
    """Async synthetic-dataset generation job (Phase 23.B).

    PHASE_23_PLAN.md §6 schema. The customer's LLM API key is encrypted
    at rest via ``pgp_sym_encrypt`` (pgcrypto extension); the symmetric
    key comes from the ``VLABS_DATA_LLM_KEY_ENCRYPTION`` env. Dataset
    payload itself lives in R2 (per D3-B ruling); this row stores
    aggregate stats only (per D9-C ruling).
    """

    __tablename__ = "dataset_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    env_id: Mapped[str] = mapped_column(Text, nullable=False)
    env_version: Mapped[str] = mapped_column(Text, nullable=False)
    requested_tuples: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_tuples: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    seed_start: Mapped[int] = mapped_column(BigInteger, nullable=False)
    seed_end: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Customer-supplied LLM endpoint config.
    llm_endpoint_url: Mapped[str] = mapped_column(Text, nullable=False)
    llm_api_key_encrypted: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    llm_model: Mapped[str] = mapped_column(Text, nullable=False)

    budget_usd_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    budget_usd_spent: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    # State machine (PHASE_23_PLAN.md §9).
    state: Mapped[str] = mapped_column(Text, nullable=False, default="created")
    output_format: Mapped[str] = mapped_column(
        Text, nullable=False, default="parquet"
    )

    # Aggregate stats (D9-C). NULL until completion.
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    p25_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    p50_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    p75_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    completion_success_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Storage pointer (R2 object key + integrity).
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )

    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    hard_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("dataset_jobs_user_idx", "user_id", "created_at"),
        Index("dataset_jobs_state_idx", "state", "created_at"),
        Index(
            "dataset_jobs_idempotency_idx",
            "idempotency_key",
            "user_id",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class Monitor(Base):
    """Continuous-capability monitor configuration (Phase 28.B).

    PHASE_28_PLAN.md §6 schema. The customer-supplied LLM auth token
    is encrypted at rest via the existing Fernet helper
    (:mod:`vlabs_api.llm_key_crypto`); the ``auth_token_fingerprint``
    is the first 8 hex chars of SHA-256(plaintext) — surfaced as the
    "is this the key I think it is?" UX without leaking the token.

    Cadence semantics (D3-A): one of ``daily | weekly | monthly``.
    The scheduler tick (Phase 28.C) reads ``next_run_at`` to decide
    when to fire. ``status='paused'`` rows are skipped; ``failed``
    rows wait for explicit re-activation.
    """

    __tablename__ = "monitors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    model_endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    auth_token_encrypted: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    auth_token_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    cadence: Mapped[str] = mapped_column(Text, nullable=False)
    env_subset: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    episodes_per_env: Mapped[int] = mapped_column(Integer, nullable=False)
    alert_channels: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    # The FK from monitors.baseline_run_id → monitor_runs.id creates a
    # circular dependency at the SQLAlchemy metadata level (monitor_runs
    # already FKs back to monitors). The Alembic migration adds the FK
    # via a separate ALTER TABLE statement after both tables exist (see
    # 0005_add_monitors.py) — at the ORM level we keep this as a plain
    # UUID column so Base.metadata.create_all/drop_all in tests doesn't
    # trip on the cycle.
    baseline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    retention_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=90
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "cadence IN ('daily','weekly','monthly')",
            name="monitors_cadence_check",
        ),
        CheckConstraint(
            "status IN ('active','paused','failed')",
            name="monitors_status_check",
        ),
        Index("monitors_user_idx", "user_id", "created_at"),
        Index(
            "monitors_next_run_idx",
            "next_run_at",
            postgresql_where=text("status='active'"),
        ),
    )


class MonitorRun(Base):
    """One firing instance of a monitor (Phase 28.B).

    Created by the scheduler tick (Phase 28.C); transitions
    ``queued → running → success | failed``. The
    ``UNIQUE(monitor_id, scheduled_at)`` constraint blocks
    duplicate-enqueue races (R10 mitigation).
    """

    __tablename__ = "monitor_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    monitor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("monitors.id", ondelete="CASCADE"),
        nullable=False,
    )
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    summary_stats: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    regression_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    pdf_storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_sha256: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd_estimate: Mapped[float | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger: Mapped[str] = mapped_column(
        Text, nullable=False, default="scheduled"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','success','failed')",
            name="monitor_runs_status_check",
        ),
        CheckConstraint(
            "regression_verdict IS NULL OR "
            "regression_verdict IN ('ok','warning','regressed')",
            name="monitor_runs_verdict_check",
        ),
        CheckConstraint(
            "trigger IN ('scheduled','manual')",
            name="monitor_runs_trigger_check",
        ),
        UniqueConstraint(
            "monitor_id", "scheduled_at",
            name="monitor_runs_idempotency",
        ),
        Index("monitor_runs_monitor_idx", "monitor_id", "scheduled_at"),
        Index("monitor_runs_status_idx", "status", "scheduled_at"),
    )


class MonitorAlert(Base):
    """Per-channel alert delivery row (Phase 28.D)."""

    __tablename__ = "monitor_alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    monitor_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("monitor_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    channel: Mapped[str] = mapped_column(Text, nullable=False)
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivery_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint(
            "channel IN ('email','slack','webhook')",
            name="monitor_alerts_channel_check",
        ),
        Index("monitor_alerts_run_idx", "monitor_run_id"),
    )


class StripeEvent(Base):
    __tablename__ = "stripe_events"

    event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    stripe_subscription_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    current_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    current_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    cancel_at_period_end: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    user: Mapped[User] = relationship(back_populates="subscriptions")

    __table_args__ = (
        CheckConstraint("tier IN ('pro','team')", name="subscriptions_tier_check"),
        Index("subscriptions_user_idx", "user_id"),
    )


# ── Engine + session factory ──────────────────────────────────────


_engine: AsyncEngine | None = None
_SessionFactory: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str) -> AsyncEngine:
    """Create the global async engine + session factory.

    Idempotent: re-calling with the same URL returns the existing
    engine. Test fixtures use :func:`override_engine` instead.

    ``statement_cache_size=0`` is required when ``database_url`` points
    at a pgbouncer-style pooler in *transaction* mode (Supabase
    Transaction Pooler is one): asyncpg's default prepared-statement
    cache collides on pooled connections that get reused across
    statements. On direct connections (pgserver in tests, Postgres
    on localhost) this is a small perf hit and behaviour-equivalent.
    """
    global _engine, _SessionFactory
    if _engine is not None and str(_engine.url) == database_url:
        return _engine
    _engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"statement_cache_size": 0},
    )
    _SessionFactory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def override_engine(engine: AsyncEngine) -> None:
    """Replace the global engine — used by test fixtures."""
    global _engine, _SessionFactory
    _engine = engine
    _SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def dispose_engine() -> None:
    global _engine, _SessionFactory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _SessionFactory = None


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an :class:`AsyncSession`."""
    if _SessionFactory is None:
        raise RuntimeError(
            "Database engine not initialised. Call init_engine() during app lifespan."
        )
    async with _SessionFactory() as session:
        yield session


__all__ = [
    "Base",
    "User",
    "APIKey",
    "CalibrationRun",
    "Evaluation",
    "UsageCounter",
    "AuditCall",
    "DatasetJob",
    "Monitor",
    "MonitorRun",
    "MonitorAlert",
    "StripeEvent",
    "Subscription",
    "init_engine",
    "override_engine",
    "dispose_engine",
    "get_db",
]
