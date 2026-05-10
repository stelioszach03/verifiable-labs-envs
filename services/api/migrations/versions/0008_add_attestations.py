"""attestations / attestation_artifacts / attestation_audits /
attestation_renewals + attestation_public view + usage_counters
extension (Phase 31.B).

Adds the four tables backing the V-Certified attestation programme +
the read-only public verification view + the per-month per-key
verification counter. Schema mirrors :doc:`PHASE_31_PLAN.md` §6.

Lifecycle status (``attestations.status``) is constrained to
``draft | submitted | under_review | approved | revoked | expired |
withdrawn`` per D3; tier to ``bronze | silver | gold`` per D4-B;
scope_type to ``model | deployment | organization`` per D1-D; cycle
to ``annual | continuous`` per D3-D.

Customer artifacts land in R2 (``attestation_artifacts.storage_uri``)
with SHA-256 hash for tamper detection. Sensitive artifacts opt into
Fernet encryption (``encrypted=true``) per D9 + R11. The
``attestation_audits`` row is the multi-party-approval trail per D12
+ R6: each row records auditor identity + decision + summary.

Renewal idempotency (D3-B) uses the existing Phase 23 partial unique
index pattern keyed on ``(idempotency_key, attestation_id)``.

Revision ID: 0008
Revises: 0007
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── attestations ──────────────────────────────────────────────
    op.create_table(
        "attestations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("public_id", sa.Text(), nullable=False, unique=True),
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
        sa.Column("organization", sa.Text(), nullable=False),
        sa.Column("scope_type", sa.Text(), nullable=False),
        sa.Column("scope_subject", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default="draft"
        ),
        sa.Column("cycle", sa.Text(), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
        sa.Column("cert_serial", sa.Text(), nullable=True, unique=True),
        sa.Column(
            "standards_alignment",
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
        sa.CheckConstraint(
            "scope_type IN ('model','deployment','organization')",
            name="attestations_scope_type_check",
        ),
        sa.CheckConstraint(
            "tier IN ('bronze','silver','gold')",
            name="attestations_tier_check",
        ),
        sa.CheckConstraint(
            "status IN ('draft','submitted','under_review','approved',"
            "'revoked','expired','withdrawn')",
            name="attestations_status_check",
        ),
        sa.CheckConstraint(
            "cycle IN ('annual','continuous')",
            name="attestations_cycle_check",
        ),
    )
    op.create_index(
        "attestations_user_idx",
        "attestations",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "attestations_status_idx", "attestations", ["status"]
    )
    op.create_index("attestations_tier_idx", "attestations", ["tier"])
    op.create_index(
        "attestations_public_id_idx",
        "attestations",
        ["public_id"],
        unique=True,
    )

    # ── attestation_artifacts ─────────────────────────────────────
    op.create_table(
        "attestation_artifacts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "attestation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("attestations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("sha256_hash", sa.Text(), nullable=False),
        sa.Column(
            "encrypted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('training_doc','audit_report','monitor_record',"
            "'rm_record','prm_record','change_mgmt','legal_signoff',"
            "'third_party_audit')",
            name="attestation_artifacts_kind_check",
        ),
    )
    op.create_index(
        "attestation_artifacts_attestation_idx",
        "attestation_artifacts",
        ["attestation_id"],
    )

    # ── attestation_audits ────────────────────────────────────────
    op.create_table(
        "attestation_audits",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "attestation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("attestations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("auditor_kind", sa.Text(), nullable=False),
        sa.Column(
            "auditor_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("auditor_label", sa.Text(), nullable=True),
        sa.Column(
            "audit_summary", postgresql.JSONB(), nullable=False
        ),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "auditor_kind IN ('self','vlabs','third_party')",
            name="attestation_audits_auditor_kind_check",
        ),
        sa.CheckConstraint(
            "decision IN ('approve','reject','request_more','revoke')",
            name="attestation_audits_decision_check",
        ),
    )
    op.create_index(
        "attestation_audits_attestation_idx",
        "attestation_audits",
        ["attestation_id", sa.text("decided_at DESC")],
    )

    # ── attestation_renewals ──────────────────────────────────────
    op.create_table(
        "attestation_renewals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "attestation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("attestations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cycle_number", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True),
        sa.Column(
            "initiated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("new_cert_serial", sa.Text(), nullable=True),
    )
    op.create_index(
        "attestation_renewals_attestation_idx",
        "attestation_renewals",
        ["attestation_id", "cycle_number"],
    )
    op.create_index(
        "attestation_renewals_idempotency_idx",
        "attestation_renewals",
        ["idempotency_key", "attestation_id"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # ── public verification view ──────────────────────────────────
    op.execute(
        """
        CREATE VIEW attestation_public AS
        SELECT
            public_id,
            organization,
            scope_type,
            scope_subject,
            tier,
            status,
            cycle,
            issued_at,
            expires_at,
            revoked_at,
            revocation_reason,
            standards_alignment
        FROM attestations
        WHERE status IN ('approved','revoked','expired')
        """
    )

    # ── usage_counters extension ──────────────────────────────────
    op.add_column(
        "usage_counters",
        sa.Column(
            "attestation_verifications_count",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("usage_counters", "attestation_verifications_count")
    op.execute("DROP VIEW IF EXISTS attestation_public")
    op.drop_index(
        "attestation_renewals_idempotency_idx",
        table_name="attestation_renewals",
    )
    op.drop_index(
        "attestation_renewals_attestation_idx",
        table_name="attestation_renewals",
    )
    op.drop_table("attestation_renewals")
    op.drop_index(
        "attestation_audits_attestation_idx",
        table_name="attestation_audits",
    )
    op.drop_table("attestation_audits")
    op.drop_index(
        "attestation_artifacts_attestation_idx",
        table_name="attestation_artifacts",
    )
    op.drop_table("attestation_artifacts")
    op.drop_index("attestations_public_id_idx", table_name="attestations")
    op.drop_index("attestations_tier_idx", table_name="attestations")
    op.drop_index("attestations_status_idx", table_name="attestations")
    op.drop_index("attestations_user_idx", table_name="attestations")
    op.drop_table("attestations")
