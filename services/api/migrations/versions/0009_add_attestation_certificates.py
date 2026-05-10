"""attestation_certificates table for the V-Certified PKI (Phase 31.D).

Each approved attestation receives one signed X.509 leaf certificate;
the PEM lives in this table keyed by ``cert_serial`` and indexed by
``attestation_id`` for the public verification endpoints. The leaf
private key is stored briefly for customer one-time download then
zeroised by the renewal cycle (R11 deferred — v0.0.1 stores the key
PEM alongside the cert; production hardening in 31.G+ will move the
key to KMS).

Revision ID: 0009
Revises: 0008
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "attestation_certificates",
        sa.Column(
            "cert_serial",
            sa.Text(),
            primary_key=True,
        ),
        sa.Column(
            "attestation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("attestations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("certificate_pem", sa.Text(), nullable=False),
        sa.Column("private_key_pem", sa.Text(), nullable=True),
        sa.Column(
            "issued_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "revoked_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_attestation_certificates_attestation_id",
        "attestation_certificates",
        ["attestation_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_attestation_certificates_attestation_id",
        table_name="attestation_certificates",
    )
    op.drop_table("attestation_certificates")
