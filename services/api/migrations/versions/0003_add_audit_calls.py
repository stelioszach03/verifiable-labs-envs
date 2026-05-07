"""audit_calls — per-call scoring audit log + scores_count usage column.

Phase 22.B migration. Adds the ``audit_calls`` table that backs
``GET /v1/score/audit/{audit_id}`` and the per-call training-API
audit trail. Also adds a ``scores_count`` column to ``usage_counters``
so the new ``/v1/instance`` and ``/v1/score`` endpoints can be metered
without breaking the existing trace counter.

Schema and indexes mirror PHASE_22_PLAN.md §6.

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_calls",
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
        sa.Column(
            "api_key_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("env_id", sa.Text(), nullable=False),
        sa.Column("env_version", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("completion_hash", sa.Text(), nullable=False),
        sa.Column("reward", sa.Float(), nullable=False),
        sa.Column("conformal_low", sa.Float(), nullable=False),
        sa.Column("conformal_high", sa.Float(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("components_json", postgresql.JSONB(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "audit_calls_user_idx",
        "audit_calls",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "audit_calls_env_idx",
        "audit_calls",
        ["env_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "audit_calls_idempotency_idx",
        "audit_calls",
        ["idempotency_key", "user_id"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # New scores_count column on the existing usage_counters table.
    op.add_column(
        "usage_counters",
        sa.Column(
            "scores_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    op.execute("ALTER TABLE audit_calls ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_column("usage_counters", "scores_count")
    op.drop_index("audit_calls_idempotency_idx", table_name="audit_calls")
    op.drop_index("audit_calls_env_idx", table_name="audit_calls")
    op.drop_index("audit_calls_user_idx", table_name="audit_calls")
    op.drop_table("audit_calls")
