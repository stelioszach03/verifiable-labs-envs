"""initial vlabs-api schema

Mirrors the SQL migration ``0001_initial_vlabs_api_schema`` already
applied to the production Supabase project. This file lets developers
build the schema against any Postgres (Docker, pgserver) via
``alembic upgrade head``.

Revision ID: 0001
Revises:
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text()),
        sa.Column("clerk_user_id", sa.Text(), unique=True),
        sa.Column("stripe_customer_id", sa.Text(), unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key_hash", sa.LargeBinary(), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(length=8), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index("api_keys_user_idx", "api_keys", ["user_id"])

    op.create_table(
        "calibration_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("nonconformity", sa.Text(), nullable=False),
        sa.Column("n_calibration", sa.Integer(), nullable=False),
        sa.Column("quantile", sa.Float(), nullable=False),
        sa.Column("nonconformity_stats", postgresql.JSONB(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("request_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("request_traces", sa.Integer(), nullable=False),
    )
    op.create_index(
        "calibration_runs_owner_idx",
        "calibration_runs",
        ["api_key_id", "created_at"],
    )

    op.create_table(
        "evaluations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "calibration_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("calibration_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("empirical_coverage", sa.Float(), nullable=False),
        sa.Column("target_coverage", sa.Float(), nullable=False),
        sa.Column("passes", sa.Boolean(), nullable=False),
        sa.Column("tolerance", sa.Float(), nullable=False),
        sa.Column("interval_width_mean", sa.Float(), nullable=False),
        sa.Column("nonconformity_stats", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("request_traces", sa.Integer(), nullable=False),
    )
    op.create_index(
        "evaluations_calib_idx", "evaluations", ["calibration_id", "created_at"]
    )

    op.create_table(
        "usage_counters",
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("month", sa.Date(), primary_key=True),
        sa.Column("traces_count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "calibrations_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("evaluations_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("predictions_count", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "subscriptions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stripe_subscription_id", sa.Text(), nullable=False, unique=True),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "cancel_at_period_end",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "tier IN ('pro','team')", name="subscriptions_tier_check"
        ),
    )
    op.create_index("subscriptions_user_idx", "subscriptions", ["user_id"])

    for tbl in (
        "users",
        "api_keys",
        "calibration_runs",
        "evaluations",
        "usage_counters",
        "subscriptions",
    ):
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    for tbl in (
        "subscriptions",
        "usage_counters",
        "evaluations",
        "calibration_runs",
        "api_keys",
        "users",
    ):
        op.drop_table(tbl)
