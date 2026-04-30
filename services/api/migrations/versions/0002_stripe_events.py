"""stripe_events — webhook idempotency dedup table

Mirrors the SQL migration ``0002_stripe_events_dedup`` already applied
to the production Supabase project.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stripe_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
    )
    op.create_index(
        "stripe_events_type_idx", "stripe_events", ["event_type", "received_at"]
    )
    op.execute("ALTER TABLE stripe_events ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_table("stripe_events")
