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
    String,
    Text,
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
    """
    global _engine, _SessionFactory
    if _engine is not None and str(_engine.url) == database_url:
        return _engine
    _engine = create_async_engine(database_url, pool_pre_ping=True)
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
    "StripeEvent",
    "Subscription",
    "init_engine",
    "override_engine",
    "dispose_engine",
    "get_db",
]
