"""Tests for scripts/preflight/verify_migrations.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "verify_migrations.py"


def _load_module():
    """Import the script as a module without invoking ``main()``.

    Registers in ``sys.modules`` first so the dataclass machinery
    can resolve the module reference for :class:`MigrationFile`.
    """
    spec = importlib.util.spec_from_file_location("verify_migrations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vm():
    return _load_module()


# ── on-disk migration discovery ───────────────────────────────────


def test_discover_local_migrations_returns_nine(vm) -> None:
    migrations = vm.discover_local_migrations()
    assert len(migrations) == 9
    revisions = [m.revision for m in migrations]
    assert revisions == [f"{i:04d}" for i in range(1, 10)]


def test_head_revision_is_0009(vm) -> None:
    migrations = vm.discover_local_migrations()
    assert vm.head_revision(migrations) == "0009"


def test_chain_is_linear(vm) -> None:
    migrations = vm.discover_local_migrations()
    ok, reason = vm.chain_is_linear(migrations)
    assert ok is True, reason


def test_chain_is_linear_empty_list(vm) -> None:
    ok, _ = vm.chain_is_linear([])
    assert ok is True


def test_chain_detects_skip(vm) -> None:
    """Synthesise a bad chain to exercise the linearity check."""
    migrations = [
        vm.MigrationFile(path=Path("a"), revision="0001", down_revision=None),
        vm.MigrationFile(path=Path("b"), revision="0003", down_revision="0001"),
    ]
    ok, reason = vm.chain_is_linear(migrations)
    assert ok is True  # 0003's down_revision matches 0001 → linear by spec
    bad = [
        vm.MigrationFile(path=Path("a"), revision="0001", down_revision=None),
        vm.MigrationFile(path=Path("b"), revision="0002", down_revision="9999"),
    ]
    ok2, reason2 = vm.chain_is_linear(bad)
    assert ok2 is False
    assert "0002" in reason2


# ── DSN handling ───────────────────────────────────────────────────


def test_database_url_falls_back_to_sqlite_in_memory(vm, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert vm.database_url() == "sqlite:///:memory:"


def test_database_url_strips_async_driver(vm, monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://u:p@h:5432/db"
    )
    assert vm.database_url() == "postgresql://u:p@h:5432/db"


def test_redact_strips_password(vm) -> None:
    redacted = vm._redact("postgresql://user:secret@host:5432/db")
    assert "secret" not in redacted
    assert "***" in redacted
    assert "user" in redacted


def test_redact_passes_url_with_no_at_sign(vm) -> None:
    assert vm._redact("sqlite:///:memory:") == "sqlite:///:memory:"


# ── build_report end-to-end ────────────────────────────────────────


def test_build_report_against_in_memory_sqlite(vm, monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    report = vm.build_report(apply=False)
    assert report["head_revision"] == "0009"
    # In-memory SQLite has no alembic_version table → current is None
    # → all 9 migrations are pending.
    assert report["current_revision"] is None
    assert len(report["pending_revisions"]) == 9
    assert report["chain_linear"] is True
    assert report["migration_count"] == 9


def test_build_report_no_pending_when_current_matches_head(
    vm, monkeypatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def _stub_query(_url):
        return "0009"

    monkeypatch.setattr(vm, "query_current_revision", _stub_query)
    report = vm.build_report(apply=False)
    assert report["current_revision"] == "0009"
    assert report["pending_revisions"] == []


def test_main_json_mode_is_structured(vm, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = vm.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["head_revision"] == "0009"
    assert payload["chain_linear"] is True


def test_main_text_mode_prints_summary(vm, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = vm.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "head_revision: 0009" in out
    assert "chain_linear: True" in out
