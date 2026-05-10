"""process_reward_models / process_reward_model_runs +
usage_counters extension (Phase 30.E).

Adds the two tables backing the distilled PRM service + extends
``usage_counters`` with the per-month per-key PRM-score counter.
Schema mirrors :doc:`PHASE_30_PLAN.md` §6.

Lifecycle status (``process_reward_models.status``) is constrained
to ``training | available | deprecated | retired`` per D12-B; the
``process_reward_model_runs`` row records every customer score call
with SHA-256 hashes (no plaintext) for the GDPR-aligned audit trail
mirroring Phases 22 + 29.

D9-C per-step + aggregate conformal quantiles persist in the
``process_reward_models.step_conformal_quantiles`` (JSONB) +
``aggregate_conformal_quantile`` (Float) columns. ``base_rm_id``
is the FK to ``reward_models.id`` for the D13-B/C shared backbone
path; NULL for D13-A independent serving.

Revision ID: 0007
Revises: 0006
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── process_reward_models ─────────────────────────────────────
    op.create_table(
        "process_reward_models",
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
        sa.Column(
            "base_rm_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("reward_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("step_granularity", sa.Text(), nullable=False),
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
        sa.Column(
            "step_conformal_quantiles",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column(
            "aggregate_conformal_quantile",
            sa.Float(),
            nullable=True,
        ),
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
            name="process_reward_models_status_check",
        ),
        sa.CheckConstraint(
            "step_granularity IN ('per_step','per_token','per_stage')",
            name="process_reward_models_granularity_check",
        ),
    )
    op.create_index(
        "process_reward_models_family_idx",
        "process_reward_models",
        ["family", sa.text("created_at DESC")],
    )
    op.create_index(
        "process_reward_models_status_idx",
        "process_reward_models",
        ["status"],
    )
    op.create_index(
        "process_reward_models_base_rm_idx",
        "process_reward_models",
        ["base_rm_id"],
    )

    # ── process_reward_model_runs ─────────────────────────────────
    op.create_table(
        "process_reward_model_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "process_reward_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "process_reward_models.id", ondelete="CASCADE"
            ),
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
        sa.Column("trace_hash", sa.Text(), nullable=False),
        sa.Column("env_id", sa.Text(), nullable=True),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("step_rewards", postgresql.JSONB(), nullable=False),
        sa.Column("step_cis", postgresql.JSONB(), nullable=False),
        sa.Column("aggregate_reward", sa.Float(), nullable=False),
        sa.Column("aggregate_ci_low", sa.Float(), nullable=False),
        sa.Column("aggregate_ci_high", sa.Float(), nullable=False),
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
        "process_reward_model_runs_user_idx",
        "process_reward_model_runs",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "process_reward_model_runs_model_idx",
        "process_reward_model_runs",
        ["process_reward_model_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "process_reward_model_runs_idempotency_idx",
        "process_reward_model_runs",
        ["idempotency_key", "user_id"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # ── usage_counters extension ──────────────────────────────────
    op.add_column(
        "usage_counters",
        sa.Column(
            "process_reward_scores_count",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("usage_counters", "process_reward_scores_count")
    op.drop_index(
        "process_reward_model_runs_idempotency_idx",
        table_name="process_reward_model_runs",
    )
    op.drop_index(
        "process_reward_model_runs_model_idx",
        table_name="process_reward_model_runs",
    )
    op.drop_index(
        "process_reward_model_runs_user_idx",
        table_name="process_reward_model_runs",
    )
    op.drop_table("process_reward_model_runs")
    op.drop_index(
        "process_reward_models_base_rm_idx",
        table_name="process_reward_models",
    )
    op.drop_index(
        "process_reward_models_status_idx",
        table_name="process_reward_models",
    )
    op.drop_index(
        "process_reward_models_family_idx",
        table_name="process_reward_models",
    )
    op.drop_table("process_reward_models")
