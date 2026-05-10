"""Schema-shape tests for Phase 31.B Alembic migration 0008."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "services" / "api" / "migrations" / "versions"


def _load_migration():
    path = MIGRATIONS / "0008_add_attestations.py"
    spec = importlib.util.spec_from_file_location(
        "_alembic_08_attestations", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_chain_from_0007() -> None:
    m = _load_migration()
    assert m.revision == "0008"
    assert m.down_revision == "0007"
    assert m.branch_labels is None
    assert m.depends_on is None


def test_upgrade_downgrade_callable_present() -> None:
    m = _load_migration()
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_migration_creates_four_tables() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    for tbl in (
        "attestations",
        "attestation_artifacts",
        "attestation_audits",
        "attestation_renewals",
    ):
        assert f'create_table(\n        "{tbl}"' in text


def test_migration_creates_public_view() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert "CREATE VIEW attestation_public" in text
    assert "DROP VIEW IF EXISTS attestation_public" in text


def test_migration_extends_usage_counters() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert 'add_column(\n        "usage_counters"' in text
    assert "attestation_verifications_count" in text


def test_migration_creates_expected_indexes() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    for idx in (
        "attestations_user_idx",
        "attestations_status_idx",
        "attestations_tier_idx",
        "attestations_public_id_idx",
        "attestation_artifacts_attestation_idx",
        "attestation_audits_attestation_idx",
        "attestation_renewals_attestation_idx",
        "attestation_renewals_idempotency_idx",
    ):
        assert idx in text


def test_migration_status_check_constraint() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert "attestations_status_check" in text
    for status in (
        "'draft'",
        "'submitted'",
        "'under_review'",
        "'approved'",
        "'revoked'",
        "'expired'",
        "'withdrawn'",
    ):
        assert status in text


def test_migration_tier_check_constraint() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert "attestations_tier_check" in text
    assert "'bronze','silver','gold'" in text


def test_migration_scope_type_check() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert "attestations_scope_type_check" in text
    assert "'model','deployment','organization'" in text


def test_migration_cycle_check() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert "attestations_cycle_check" in text
    assert "'annual','continuous'" in text


def test_migration_artifact_kind_check_lists_all_8() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    for kind in (
        "'training_doc'",
        "'audit_report'",
        "'monitor_record'",
        "'rm_record'",
        "'prm_record'",
        "'change_mgmt'",
        "'legal_signoff'",
        "'third_party_audit'",
    ):
        assert kind in text


def test_migration_partial_unique_idempotency_index() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    assert (
        'postgresql_where=sa.text("idempotency_key IS NOT NULL")'
        in text
    )


def test_downgrade_drops_view_and_tables_in_reverse_order() -> None:
    text = (MIGRATIONS / "0008_add_attestations.py").read_text(encoding="utf-8")
    drop_col = text.index('drop_column("usage_counters"')
    drop_view = text.index("DROP VIEW IF EXISTS attestation_public")
    drop_renewals = text.index('drop_table("attestation_renewals"')
    drop_audits = text.index('drop_table("attestation_audits"')
    drop_artifacts = text.index('drop_table("attestation_artifacts"')
    drop_attestations = text.index('drop_table("attestations"')
    assert (
        drop_col
        < drop_view
        < drop_renewals
        < drop_audits
        < drop_artifacts
        < drop_attestations
    )


def test_orm_classes_match_migration() -> None:
    from vlabs_api import db as orm

    assert orm.Attestation.__tablename__ == "attestations"
    assert orm.AttestationArtifact.__tablename__ == "attestation_artifacts"
    assert orm.AttestationAudit.__tablename__ == "attestation_audits"
    assert orm.AttestationRenewal.__tablename__ == "attestation_renewals"
    for name in (
        "Attestation",
        "AttestationArtifact",
        "AttestationAudit",
        "AttestationRenewal",
    ):
        assert name in orm.__all__


def test_orm_usage_counter_has_attestation_verifications_count() -> None:
    from vlabs_api.db import UsageCounter

    cols = {c.name for c in UsageCounter.__table__.columns}
    assert "attestation_verifications_count" in cols
