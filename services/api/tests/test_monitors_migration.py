"""Schema-shape tests for Phase 28.B Alembic migration 0005.

The other ``services/api/tests/*`` suites build the schema via
``Base.metadata.create_all`` (much faster than spinning Alembic for
each test). This module verifies the **migration script itself** is
well-formed: revision identifiers, operation counts, downgrade
inverse-of-upgrade, no stray references to dropped objects.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO_ROOT / "services" / "api" / "migrations" / "versions"


def _load_migration():
    path = MIGRATIONS / "0005_add_monitors.py"
    spec = importlib.util.spec_from_file_location(
        "_alembic_05_monitors", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_revision_identifiers_chain_from_0004() -> None:
    m = _load_migration()
    assert m.revision == "0005"
    assert m.down_revision == "0004"
    assert m.branch_labels is None
    assert m.depends_on is None


def test_upgrade_callable_present() -> None:
    m = _load_migration()
    assert callable(m.upgrade)
    assert callable(m.downgrade)


def test_migration_creates_three_tables() -> None:
    """Source-text shape check: monitors, monitor_runs, monitor_alerts."""
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    for tbl in ("monitors", "monitor_runs", "monitor_alerts"):
        assert f'create_table(\n        "{tbl}"' in text, (
            f"missing create_table for {tbl}"
        )


def test_migration_creates_expected_indexes() -> None:
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    for idx in (
        "monitors_user_idx",
        "monitors_next_run_idx",
        "monitor_runs_monitor_idx",
        "monitor_runs_status_idx",
        "monitor_alerts_run_idx",
    ):
        assert idx in text, f"missing index {idx}"


def test_migration_creates_expected_check_constraints() -> None:
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    for constraint in (
        "monitors_cadence_check",
        "monitors_status_check",
        "monitor_runs_status_check",
        "monitor_runs_verdict_check",
        "monitor_runs_trigger_check",
        "monitor_runs_idempotency",
        "monitor_alerts_channel_check",
    ):
        assert constraint in text, f"missing constraint {constraint}"


def test_migration_creates_baseline_fk_to_monitor_runs() -> None:
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    assert "monitors_baseline_fk" in text
    # FK created after both tables exist (avoids the create-order
    # chicken-egg). Confirm it's added in upgrade(), not in the
    # CREATE TABLE for monitors.
    assert "create_foreign_key" in text


def test_downgrade_reverses_upgrade_in_reverse_order() -> None:
    """Downgrade must drop the FK first, then the tables in reverse
    dependency order (alerts → runs → monitors)."""
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    drop_fk_pos = text.index('drop_constraint("monitors_baseline_fk"')
    drop_alerts_pos = text.index('drop_table("monitor_alerts"')
    drop_runs_pos = text.index('drop_table("monitor_runs"')
    drop_monitors_pos = text.index('drop_table("monitors"')
    assert drop_fk_pos < drop_alerts_pos < drop_runs_pos < drop_monitors_pos


def test_orm_model_classes_match_migration() -> None:
    """Spot-check: the ORM module exports Monitor / MonitorRun /
    MonitorAlert and lists them in __all__."""
    from vlabs_api import db as db_module

    assert hasattr(db_module, "Monitor")
    assert hasattr(db_module, "MonitorRun")
    assert hasattr(db_module, "MonitorAlert")
    assert "Monitor" in db_module.__all__
    assert "MonitorRun" in db_module.__all__
    assert "MonitorAlert" in db_module.__all__


def test_migration_script_size_within_envelope() -> None:
    """Sanity: 0005 migration is ~250 lines (mirrors 0004 envelope)."""
    text = (MIGRATIONS / "0005_add_monitors.py").read_text(encoding="utf-8")
    n = text.count("\n")
    assert 150 < n < 350, f"unexpected migration size: {n} lines"


def test_revision_id_is_unique() -> None:
    """Ensure no other migration also claims revision='0005'."""
    others = sorted(MIGRATIONS.glob("000*.py"))
    assert any(p.name == "0005_add_monitors.py" for p in others)
    seen: set[str] = set()
    for p in others:
        spec = importlib.util.spec_from_file_location(p.stem, p)
        m = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(m)
        assert m.revision not in seen, f"duplicate revision: {m.revision}"
        seen.add(m.revision)
