"""monitors / monitor_runs / monitor_alerts (Phase 28.B).

Adds the three tables that back the continuous-capability monitoring
product surface — `monitors` (long-lived configurations), `monitor_runs`
(time-bounded firing instances), `monitor_alerts` (delivery audit
trail). Also extends `tier_limits` (the Postgres table populated
elsewhere) is **not** modified — Phase 28 stores its tier numbers in
``vlabs_api.config.TierLimits`` (pydantic settings), not in the DB.

Schema mirrors PHASE_28_PLAN.md §6. The Fernet-encrypted customer auth
token lives in ``monitors.auth_token_encrypted`` (BYTEA) — the
encryption key comes from ``VLABS_DATA_LLM_KEY_ENCRYPTION`` (already a
Fly secret per Phase 23.C).

Revision ID: 0005
Revises: 0004
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── monitors ────────────────────────────────────────────────────
    op.create_table(
        "monitors",
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
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("model_endpoint", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("auth_token_encrypted", postgresql.BYTEA(), nullable=False),
        sa.Column("auth_token_fingerprint", sa.Text(), nullable=False),
        sa.Column("cadence", sa.Text(), nullable=False),
        sa.Column("env_subset", postgresql.JSONB(), nullable=False),
        sa.Column("episodes_per_env", sa.Integer(), nullable=False),
        sa.Column(
            "alert_channels",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "baseline_run_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "retention_days",
            sa.Integer(),
            nullable=False,
            server_default="90",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "cadence IN ('daily','weekly','monthly')",
            name="monitors_cadence_check",
        ),
        sa.CheckConstraint(
            "status IN ('active','paused','failed')",
            name="monitors_status_check",
        ),
    )
    op.create_index(
        "monitors_user_idx",
        "monitors",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "monitors_next_run_idx",
        "monitors",
        ["next_run_at"],
        postgresql_where=sa.text("status='active'"),
    )

    # ── monitor_runs ───────────────────────────────────────────────
    op.create_table(
        "monitor_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "monitor_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("monitors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("summary_stats", postgresql.JSONB(), nullable=True),
        sa.Column("regression_verdict", sa.Text(), nullable=True),
        sa.Column("verdict_payload", postgresql.JSONB(), nullable=True),
        sa.Column("pdf_storage_key", sa.Text(), nullable=True),
        sa.Column("pdf_sha256", sa.Text(), nullable=True),
        sa.Column("cost_usd_estimate", sa.Numeric(10, 4), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "trigger",
            sa.Text(),
            nullable=False,
            server_default="scheduled",
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','success','failed')",
            name="monitor_runs_status_check",
        ),
        sa.CheckConstraint(
            "regression_verdict IS NULL OR "
            "regression_verdict IN ('ok','warning','regressed')",
            name="monitor_runs_verdict_check",
        ),
        sa.CheckConstraint(
            "trigger IN ('scheduled','manual')",
            name="monitor_runs_trigger_check",
        ),
        sa.UniqueConstraint(
            "monitor_id",
            "scheduled_at",
            name="monitor_runs_idempotency",
        ),
    )
    op.create_index(
        "monitor_runs_monitor_idx",
        "monitor_runs",
        ["monitor_id", sa.text("scheduled_at DESC")],
    )
    op.create_index(
        "monitor_runs_status_idx",
        "monitor_runs",
        ["status", "scheduled_at"],
    )

    # ── monitor_alerts ─────────────────────────────────────────────
    op.create_table(
        "monitor_alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "monitor_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("monitor_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_error", sa.Text(), nullable=True),
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.CheckConstraint(
            "channel IN ('email','slack','webhook')",
            name="monitor_alerts_channel_check",
        ),
    )
    op.create_index(
        "monitor_alerts_run_idx",
        "monitor_alerts",
        ["monitor_run_id"],
    )

    # FK from monitors.baseline_run_id → monitor_runs (after both
    # tables exist to avoid the create-order chicken-egg).
    op.create_foreign_key(
        "monitors_baseline_fk",
        "monitors",
        "monitor_runs",
        ["baseline_run_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("monitors_baseline_fk", "monitors", type_="foreignkey")
    op.drop_index("monitor_alerts_run_idx", table_name="monitor_alerts")
    op.drop_table("monitor_alerts")
    op.drop_index("monitor_runs_status_idx", table_name="monitor_runs")
    op.drop_index("monitor_runs_monitor_idx", table_name="monitor_runs")
    op.drop_table("monitor_runs")
    op.drop_index("monitors_next_run_idx", table_name="monitors")
    op.drop_index("monitors_user_idx", table_name="monitors")
    op.drop_table("monitors")
