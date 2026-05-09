"""reward_models / reward_model_runs + usage_counters extension (Phase 29.E).

Adds the two tables backing the distilled reward model service +
extends ``usage_counters`` with the per-month per-key score counter.
Schema mirrors :doc:`PHASE_29_PLAN.md` §6.

Lifecycle status (``reward_models.status``) is constrained to
``training | available | deprecated | retired`` per D12-B; the
``reward_model_runs`` row records every customer score call with
SHA-256 hashes (no plaintext) for the GDPR-aligned audit trail.

Revision ID: 0006
Revises: 0005
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── reward_models ──────────────────────────────────────────────
    op.create_table(
        "reward_models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("model_id", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("family", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("teacher_source", sa.Text(), nullable=False),
        sa.Column("student_arch", sa.Text(), nullable=False),
        sa.Column("training_method", sa.Text(), nullable=False),
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("dataset_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("checkpoint_uri", sa.Text(), nullable=True),
        sa.Column("conformal_quantile", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="training",
        ),
        sa.Column("eval_metrics", postgresql.JSONB(), nullable=True),
        sa.Column(
            "training_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('training','available','deprecated','retired')",
            name="reward_models_status_check",
        ),
    )
    op.create_index(
        "reward_models_family_idx",
        "reward_models",
        ["family", sa.text("created_at DESC")],
    )
    op.create_index(
        "reward_models_status_idx",
        "reward_models",
        ["status"],
    )

    # ── reward_model_runs ──────────────────────────────────────────
    op.create_table(
        "reward_model_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "reward_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reward_models.id", ondelete="CASCADE"),
            nullable=False,
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
        sa.Column("prompt_hash", sa.Text(), nullable=False),
        sa.Column("response_hash", sa.Text(), nullable=False),
        sa.Column("env_id", sa.Text(), nullable=True),
        sa.Column("reward_score", sa.Float(), nullable=False),
        sa.Column("ci_low", sa.Float(), nullable=False),
        sa.Column("ci_high", sa.Float(), nullable=False),
        sa.Column(
            "coverage_guarantee",
            sa.Float(),
            nullable=False,
            server_default="0.9",
        ),
        sa.Column(
            "cache_hit",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "reward_model_runs_user_idx",
        "reward_model_runs",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "reward_model_runs_model_idx",
        "reward_model_runs",
        ["reward_model_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "reward_model_runs_idempotency_idx",
        "reward_model_runs",
        ["idempotency_key", "user_id"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # ── usage_counters extension ───────────────────────────────────
    op.add_column(
        "usage_counters",
        sa.Column(
            "reward_scores_count",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("usage_counters", "reward_scores_count")
    op.drop_index(
        "reward_model_runs_idempotency_idx", table_name="reward_model_runs"
    )
    op.drop_index("reward_model_runs_model_idx", table_name="reward_model_runs")
    op.drop_index("reward_model_runs_user_idx", table_name="reward_model_runs")
    op.drop_table("reward_model_runs")
    op.drop_index("reward_models_status_idx", table_name="reward_models")
    op.drop_index("reward_models_family_idx", table_name="reward_models")
    op.drop_table("reward_models")
