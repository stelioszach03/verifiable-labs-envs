"""scripts/preflight/verify_migrations.py — alembic migration preflight.

Connects to ``DATABASE_URL`` (sync DSN), reads the current alembic
revision via ``alembic_version``, compares against the on-disk
migration head, and reports whether the deploy will need to run
migrations.

Designed to be safe to run against production:
- Read-only by default (use ``--apply`` to actually run upgrades).
- Falls back to in-memory SQLite when ``DATABASE_URL`` is unset, so
  tests + CI can exercise the path without a live Postgres.

Usage:
    python scripts/preflight/verify_migrations.py
    python scripts/preflight/verify_migrations.py --apply
    python scripts/preflight/verify_migrations.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "services" / "api" / "migrations" / "versions"
ALEMBIC_INI = REPO_ROOT / "services" / "api" / "alembic.ini"


@dataclass(frozen=True)
class MigrationFile:
    """One on-disk Alembic revision file."""

    path: Path
    revision: str
    down_revision: str | None


def discover_local_migrations(
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[MigrationFile]:
    """Parse every ``NNNN_*.py`` file under ``migrations_dir`` and
    return them in revision order. Robust to both quoted-string and
    bare-int revision identifiers."""
    out: list[MigrationFile] = []
    if not migrations_dir.is_dir():
        return out
    rev_re = re.compile(
        r'^revision[^=]*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE
    )
    down_re = re.compile(
        r'^down_revision[^=]*=\s*([\'"]([^\'"]+)[\'"]|None)',
        re.MULTILINE,
    )
    for f in sorted(migrations_dir.glob("[0-9]*_*.py")):
        text = f.read_text(encoding="utf-8")
        rev_m = rev_re.search(text)
        if not rev_m:
            continue
        down_m = down_re.search(text)
        down: str | None = None
        if down_m and down_m.group(2):
            down = down_m.group(2)
        out.append(
            MigrationFile(path=f, revision=rev_m.group(1), down_revision=down)
        )
    out.sort(key=lambda mf: mf.revision)
    return out


def head_revision(migrations: list[MigrationFile]) -> str | None:
    """The highest-numbered revision is the head."""
    if not migrations:
        return None
    return migrations[-1].revision


def chain_is_linear(migrations: list[MigrationFile]) -> tuple[bool, str]:
    """Verify each migration's down_revision matches the prior file's
    revision. Returns ``(ok, reason)``."""
    for i, mf in enumerate(migrations):
        if i == 0:
            if mf.down_revision is not None:
                return False, f"first migration {mf.revision} has down_revision={mf.down_revision}"
        else:
            prev = migrations[i - 1]
            if mf.down_revision != prev.revision:
                return (
                    False,
                    f"migration {mf.revision} down_revision={mf.down_revision} "
                    f"but expected {prev.revision}",
                )
    return True, "linear chain"


def database_url() -> str:
    """Return the configured DSN, defaulting to in-memory sqlite when
    no DATABASE_URL is set (test + dev fallback)."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        # Convert async DSN ('postgresql+asyncpg://') to sync for alembic.
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return url
    return "sqlite:///:memory:"


def query_current_revision(url: str) -> str | None:
    """Return the live alembic_version row from the configured DSN, or
    None when the table doesn't exist (= fresh DB)."""
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:  # noqa: BLE001
        return f"<sqlalchemy import failed: {exc!r}>"
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            try:
                row = conn.execute(
                    text("SELECT version_num FROM alembic_version")
                ).first()
            except Exception:
                return None
            return row[0] if row else None
    finally:
        engine.dispose()


def build_report(*, apply: bool = False) -> dict[str, Any]:
    """Construct the JSON report. ``apply=True`` actually runs the
    upgrade after reporting; the default is read-only."""
    migrations = discover_local_migrations()
    head = head_revision(migrations)
    linear_ok, linear_reason = chain_is_linear(migrations)

    url = database_url()
    current: str | None
    error: str | None = None
    try:
        current = query_current_revision(url)
    except Exception as exc:  # noqa: BLE001
        current = None
        error = f"{type(exc).__name__}: {exc}"

    pending: list[str] = []
    if head is not None and current != head:
        # Pending = every migration after `current` up to head.
        seen_current = current is None
        for mf in migrations:
            if seen_current:
                pending.append(mf.revision)
            elif mf.revision == current:
                seen_current = True

    report = {
        "database_url_redacted": _redact(url),
        "head_revision": head,
        "current_revision": current,
        "pending_revisions": pending,
        "chain_linear": linear_ok,
        "chain_reason": linear_reason,
        "migration_count": len(migrations),
        "apply": apply,
        "error": error,
    }

    if apply and pending:
        try:
            _run_alembic_upgrade(url)
            report["applied"] = True
            report["current_revision"] = head
            report["pending_revisions"] = []
        except Exception as exc:  # noqa: BLE001
            report["applied"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _redact(url: str) -> str:
    """Strip the password from a DSN before logging."""
    if "@" not in url:
        return url
    head, tail = url.rsplit("@", 1)
    if "://" in head and ":" in head.split("://", 1)[1]:
        scheme, rest = head.split("://", 1)
        user = rest.split(":", 1)[0]
        return f"{scheme}://{user}:***@{tail}"
    return url


def _run_alembic_upgrade(url: str) -> None:
    """Invoke ``alembic upgrade head`` against the supplied DSN."""
    if not ALEMBIC_INI.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {ALEMBIC_INI}")
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option(
        "script_location",
        str(MIGRATIONS_DIR.parent),
    )
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run alembic upgrade head (default: read-only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text.",
    )
    ns = parser.parse_args(argv)

    report = build_report(apply=ns.apply)

    if ns.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DATABASE_URL: {report['database_url_redacted']}")
        print(f"head_revision: {report['head_revision']}")
        print(f"current_revision: {report['current_revision']}")
        print(f"pending: {report['pending_revisions'] or 'none'}")
        print(f"chain_linear: {report['chain_linear']}  ({report['chain_reason']})")
        if report.get("error"):
            print(f"error: {report['error']}")
        if ns.apply:
            print(f"applied: {report.get('applied')}")

    if report.get("error") and not ns.apply:
        return 1
    if ns.apply and report.get("applied") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
