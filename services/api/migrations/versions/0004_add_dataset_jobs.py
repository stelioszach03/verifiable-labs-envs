"""dataset_jobs — async synthetic dataset generation jobs (Phase 23.B).

Adds the ``dataset_jobs`` table that backs ``POST /v1/datasets`` and
the per-job audit summary stats (D9-C ruling: aggregate stats only,
per-tuple data lives in R2). Also adds a ``tuples_generated`` column
to ``usage_counters`` for the new vlabs-data quota counter (D8 ruling).

Schema mirrors PHASE_23_PLAN.md §6.

Revision ID: 0004
Revises: 0003
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_jobs",
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
        sa.Column("requested_tuples", sa.Integer(), nullable=False),
        sa.Column(
            "generated_tuples",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("seed_start", sa.BigInteger(), nullable=False),
        sa.Column("seed_end", sa.BigInteger(), nullable=False),
        # Customer-supplied LLM endpoint (D1-B ruling). API key is
        # encrypted at rest via pgcrypto pgp_sym_encrypt; symmetric key
        # comes from VLABS_DATA_LLM_KEY_ENCRYPTION env (Fly secret).
        sa.Column("llm_endpoint_url", sa.Text(), nullable=False),
        sa.Column("llm_api_key_encrypted", postgresql.BYTEA(), nullable=False),
        sa.Column("llm_model", sa.Text(), nullable=False),
        sa.Column("budget_usd_cap", sa.Float(), nullable=True),
        sa.Column(
            "budget_usd_spent",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "state",
            sa.Text(),
            nullable=False,
            server_default="created",
        ),
        sa.Column(
            "output_format",
            sa.Text(),
            nullable=False,
            server_default="parquet",
        ),
        # Aggregate stats (D9-C). Populated on completion.
        sa.Column("mean_reward", sa.Float(), nullable=True),
        sa.Column("std_reward", sa.Float(), nullable=True),
        sa.Column("p25_reward", sa.Float(), nullable=True),
        sa.Column("p50_reward", sa.Float(), nullable=True),
        sa.Column("p75_reward", sa.Float(), nullable=True),
        sa.Column("completion_success_rate", sa.Float(), nullable=True),
        # Storage pointer (R2 object key + integrity).
        sa.Column("storage_key", sa.Text(), nullable=True),
        sa.Column("storage_sha256", sa.Text(), nullable=True),
        sa.Column("storage_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hard_deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('created','queued','running','succeeded','failed',"
            "'archived','hard_deleted')",
            name="dataset_jobs_state_check",
        ),
    )
    op.create_index(
        "dataset_jobs_user_idx",
        "dataset_jobs",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "dataset_jobs_state_idx",
        "dataset_jobs",
        ["state", "created_at"],
    )
    op.create_index(
        "dataset_jobs_idempotency_idx",
        "dataset_jobs",
        ["idempotency_key", "user_id"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # New tuples_generated column on usage_counters (D8 ruling).
    op.add_column(
        "usage_counters",
        sa.Column(
            "tuples_generated",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )

    op.execute("ALTER TABLE dataset_jobs ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_column("usage_counters", "tuples_generated")
    op.drop_index("dataset_jobs_idempotency_idx", table_name="dataset_jobs")
    op.drop_index("dataset_jobs_state_idx", table_name="dataset_jobs")
    op.drop_index("dataset_jobs_user_idx", table_name="dataset_jobs")
    op.drop_table("dataset_jobs")
