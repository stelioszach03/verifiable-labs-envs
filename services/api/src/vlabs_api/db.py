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
    # Phase 29.E — counts /v1/reward-models/{id}/score against the per-tier
    # reward_scores_per_month cap. Idempotent re-issues do NOT increment.
    reward_scores_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    # Phase 30.E — counts /v1/process-reward-models/{id}/score against
    # the per-tier process_reward_scores_per_month cap. Idempotent re-
    # issues do NOT increment.
    process_reward_scores_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )
    # Phase 31.B — counts /v1/attestations/verify/* public verification
    # calls. Public endpoints are unauthenticated so the per-key counter
    # increments only when the verifying request carries a Vlabs key
    # (e.g. a customer's compliance pipeline running attestation checks
    # via their server-side credentials). Anonymous verifications go
    # through Cloudflare + Redis caching and are not metered per-user.
    attestation_verifications_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )


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


class RewardModel(Base):
    """Distilled reward model row (Phase 29.E).

    PHASE_29_PLAN.md §6 schema. ``model_id`` is the locked
    ``vlabs-reward-{family}-v{semver}`` shape (D12-B). ``status`` is
    constrained to ``training | available | deprecated | retired``;
    customer-facing endpoints surface only ``available`` + ``deprecated``
    rows. ``conformal_quantile`` is NULL until the calibration step
    (D10-A) lands in Phase 29.G; until then the service serves stub
    responses with ``schema_version="v0.1.0-stub"``.
    """

    __tablename__ = "reward_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    family: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    teacher_source: Mapped[str] = mapped_column(Text, nullable=False)
    student_arch: Mapped[str] = mapped_column(Text, nullable=False)
    training_method: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    checkpoint_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    conformal_quantile: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="training"
    )
    eval_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    training_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('training','available','deprecated','retired')",
            name="reward_models_status_check",
        ),
        Index("reward_models_family_idx", "family", "created_at"),
        Index("reward_models_status_idx", "status"),
    )


class RewardModelRun(Base):
    """Per-call audit row written by ``POST /v1/reward-models/{id}/score``
    (Phase 29.E).

    Mirrors the Phase 22 ``audit_calls`` GDPR posture: prompt + response
    are NEVER persisted, only their SHA-256 hashes (D11-C). Customers
    can verify a row matches their inputs by re-hashing locally;
    nobody else can recover the text.
    """

    __tablename__ = "reward_model_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    reward_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reward_models.id", ondelete="CASCADE"),
        nullable=False,
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
    prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    response_hash: Mapped[str] = mapped_column(Text, nullable=False)
    env_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    reward_score: Mapped[float] = mapped_column(Float, nullable=False)
    ci_low: Mapped[float] = mapped_column(Float, nullable=False)
    ci_high: Mapped[float] = mapped_column(Float, nullable=False)
    coverage_guarantee: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.9
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        Index("reward_model_runs_user_idx", "user_id", "created_at"),
        Index("reward_model_runs_model_idx", "reward_model_id", "created_at"),
        Index(
            "reward_model_runs_idempotency_idx",
            "idempotency_key",
            "user_id",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class ProcessRewardModel(Base):
    """Distilled process reward model row (Phase 30.E).

    PHASE_30_PLAN.md §6 schema. ``model_id`` is the locked
    ``vlabs-prm-{family}-v{semver}`` shape (D12-B). ``base_rm_id`` is
    the FK to the Phase 29 distilled outcome RM under the D13-B/C
    shared-backbone path; NULL for D13-A independent serving.
    ``step_granularity`` constrained to ``per_step | per_token |
    per_stage``; v0.0.1 ships ``per_step`` only (D1-B).
    ``step_conformal_quantiles`` is a JSONB dict keyed by step-position
    bucket label (e.g. ``"range(0, 1)"``); ``aggregate_conformal_quantile``
    is the trace-level scalar (D9-C). Both NULL until the calibration
    step (30.F) lands.
    """

    __tablename__ = "process_reward_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    family: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    base_rm_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reward_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    step_granularity: Mapped[str] = mapped_column(Text, nullable=False)
    teacher_source: Mapped[str] = mapped_column(Text, nullable=False)
    student_arch: Mapped[str] = mapped_column(Text, nullable=False)
    training_method: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    checkpoint_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_conformal_quantiles: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    aggregate_conformal_quantile: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="training"
    )
    eval_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    training_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('training','available','deprecated','retired')",
            name="process_reward_models_status_check",
        ),
        CheckConstraint(
            "step_granularity IN ('per_step','per_token','per_stage')",
            name="process_reward_models_granularity_check",
        ),
        Index("process_reward_models_family_idx", "family", "created_at"),
        Index("process_reward_models_status_idx", "status"),
        Index("process_reward_models_base_rm_idx", "base_rm_id"),
    )


class ProcessRewardModelRun(Base):
    """Per-call audit row written by
    ``POST /v1/process-reward-models/{id}/score`` (Phase 30.E).

    Mirrors the Phase 22 + 29 GDPR posture: prompt + trace are NEVER
    persisted, only their SHA-256 hashes (D11 / R11). ``step_rewards``
    + ``step_cis`` are JSONB arrays of floats and [low, high] pairs
    respectively, length = ``step_count``. Customers can verify a row
    matches their inputs by re-hashing locally; nobody else can
    recover the text.
    """

    __tablename__ = "process_reward_model_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    process_reward_model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("process_reward_models.id", ondelete="CASCADE"),
        nullable=False,
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
    prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    trace_hash: Mapped[str] = mapped_column(Text, nullable=False)
    env_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    step_rewards: Mapped[list[float]] = mapped_column(JSONB, nullable=False)
    step_cis: Mapped[list[list[float]]] = mapped_column(JSONB, nullable=False)
    aggregate_reward: Mapped[float] = mapped_column(Float, nullable=False)
    aggregate_ci_low: Mapped[float] = mapped_column(Float, nullable=False)
    aggregate_ci_high: Mapped[float] = mapped_column(Float, nullable=False)
    coverage_guarantee: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.9
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        Index("process_reward_model_runs_user_idx", "user_id", "created_at"),
        Index(
            "process_reward_model_runs_model_idx",
            "process_reward_model_id",
            "created_at",
        ),
        Index(
            "process_reward_model_runs_idempotency_idx",
            "idempotency_key",
            "user_id",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class Attestation(Base):
    """V-Certified attestation record (Phase 31.B).

    PHASE_31_PLAN.md §6 schema. ``public_id`` is the short URL-safe
    ``vl-<8-base32>`` identifier surfaced on the public verification
    endpoint; ``id`` is the internal UUID. ``cert_serial`` is the
    locked X.509 serial issued by the V-Certified Intermediate CA
    (D5-A) on approval; NULL until issued + after revocation.
    ``standards_alignment`` snapshots the crosswalk versions used at
    issuance time (R1 mitigation — frozen for the lifetime of this
    attestation regardless of upstream framework revisions).
    """

    __tablename__ = "attestations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    public_id: Mapped[str] = mapped_column(
        Text, nullable=False, unique=True
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
    organization: Mapped[str] = mapped_column(Text, nullable=False)
    scope_type: Mapped[str] = mapped_column(Text, nullable=False)
    scope_subject: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="draft"
    )
    cycle: Mapped[str] = mapped_column(Text, nullable=False)
    issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revocation_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    cert_serial: Mapped[str | None] = mapped_column(
        Text, nullable=True, unique=True
    )
    standards_alignment: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        CheckConstraint(
            "scope_type IN ('model','deployment','organization')",
            name="attestations_scope_type_check",
        ),
        CheckConstraint(
            "tier IN ('bronze','silver','gold')",
            name="attestations_tier_check",
        ),
        CheckConstraint(
            "status IN ('draft','submitted','under_review','approved',"
            "'revoked','expired','withdrawn')",
            name="attestations_status_check",
        ),
        CheckConstraint(
            "cycle IN ('annual','continuous')",
            name="attestations_cycle_check",
        ),
        Index("attestations_user_idx", "user_id", "created_at"),
        Index("attestations_status_idx", "status"),
        Index("attestations_tier_idx", "tier"),
    )


class AttestationArtifact(Base):
    """One supporting evidence artifact for an attestation (Phase 31.B).

    ``kind`` is the locked D9 enumeration. ``storage_uri`` points to
    R2 (``r2://vlabs-attestations/<attestation_id>/<artifact_id>/``);
    ``sha256_hash`` is computed at upload for tamper detection.
    ``encrypted=true`` opts into Fernet encryption for sensitive
    customer trade-secret artifacts (R11 mitigation).
    """

    __tablename__ = "attestation_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    attestation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("attestations.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(Text, nullable=False)
    encrypted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('training_doc','audit_report','monitor_record',"
            "'rm_record','prm_record','change_mgmt','legal_signoff',"
            "'third_party_audit')",
            name="attestation_artifacts_kind_check",
        ),
        Index("attestation_artifacts_attestation_idx", "attestation_id"),
    )


class AttestationAudit(Base):
    """Multi-party audit decision row for an attestation (Phase 31.B).

    Each row is one auditor's decision recorded as part of the D12 +
    R6 multi-party-approval trail. ``auditor_kind`` matches D10-D
    tier mapping (``self`` for Bronze, ``vlabs`` for Silver,
    ``third_party`` for Gold). ``audit_summary`` is a JSONB structured
    record (artifact-by-artifact verdict + free-form notes).
    Multi-signature revocation under §5 D12 condition 1 requires
    ≥ 2 ``auditor_kind='vlabs'`` rows with ``decision='revoke'``.
    """

    __tablename__ = "attestation_audits"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    attestation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("attestations.id", ondelete="CASCADE"),
        nullable=False,
    )
    auditor_kind: Mapped[str] = mapped_column(Text, nullable=False)
    auditor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    auditor_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    audit_summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False
    )
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        CheckConstraint(
            "auditor_kind IN ('self','vlabs','third_party')",
            name="attestation_audits_auditor_kind_check",
        ),
        CheckConstraint(
            "decision IN ('approve','reject','request_more','revoke')",
            name="attestation_audits_decision_check",
        ),
        Index(
            "attestation_audits_attestation_idx",
            "attestation_id",
            "decided_at",
        ),
    )


class AttestationRenewal(Base):
    """One renewal cycle initiation for an attestation (Phase 31.B).

    Records the lifecycle of a recertification (D3-B annual + D3-C
    continuous monthly check). ``cycle_number`` is monotonic per
    attestation. ``idempotency_key`` lets the client safely retry
    ``POST /v1/attestations/{id}/renew`` within a 24 h window via
    the same partial-unique-index pattern as Phase 23.
    """

    __tablename__ = "attestation_renewals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    attestation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("attestations.id", ondelete="CASCADE"),
        nullable=False,
    )
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    initiated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    new_cert_serial: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    __table_args__ = (
        Index(
            "attestation_renewals_attestation_idx",
            "attestation_id",
            "cycle_number",
        ),
        Index(
            "attestation_renewals_idempotency_idx",
            "idempotency_key",
            "attestation_id",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )


class AttestationCertificate(Base):
    """Issued V-Certified leaf certificate (Phase 31.D).

    One row per approved attestation cert. The PEM is served by the
    public verification endpoint (``GET /v1/attestations/verify/{id}``)
    so verifiers can chain it to the V-Certified CA without extra
    round-trips. The leaf private key is retained briefly for one-time
    customer download; production hardening (31.G+) will move it to
    KMS.
    """

    __tablename__ = "attestation_certificates"

    cert_serial: Mapped[str] = mapped_column(Text, primary_key=True)
    attestation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("attestations.id", ondelete="CASCADE"),
        nullable=False,
    )
    certificate_pem: Mapped[str] = mapped_column(Text, nullable=False)
    private_key_pem: Mapped[str | None] = mapped_column(Text, nullable=True)
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index(
            "ix_attestation_certificates_attestation_id",
            "attestation_id",
        ),
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
    "RewardModel",
    "RewardModelRun",
    "ProcessRewardModel",
    "ProcessRewardModelRun",
    "Attestation",
    "AttestationArtifact",
    "AttestationAudit",
    "AttestationCertificate",
    "AttestationRenewal",
    "StripeEvent",
    "Subscription",
    "init_engine",
    "override_engine",
    "dispose_engine",
    "get_db",
]
