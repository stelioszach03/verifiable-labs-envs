"""Local SQLite store for audit runs.

Schema (see ``_SCHEMA``):

* ``audits``: one row per ``vlabs-audit audit`` invocation. Captures the
  resolved config so runs are reproducible long after the fact.
* ``audit_runs``: one row per (env, episode) pair. State machine:
  ``pending → running → success | failed``.

Thread-safety: a single ``sqlite3.Connection`` is shared across worker
threads (``check_same_thread=False``) and serialised by an internal
``threading.Lock``. Auto-commit (``isolation_level=None``) — every
statement is its own transaction, so a crashed worker never leaves a
half-committed audit_run row.

Default DB location: ``$VLABS_AUDIT_HOME/audits.db`` where
``VLABS_AUDIT_HOME`` defaults to ``~/.vlabs-audit``. Tests pass an
explicit ``db_path`` (often a ``tmp_path``) so they never touch the
user's real cache.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id TEXT NOT NULL REFERENCES audits(id) ON DELETE CASCADE,
    env TEXT NOT NULL,
    episode_idx INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    jsonl_path TEXT,
    reward REAL,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (audit_id, env, episode_idx)
);

CREATE INDEX IF NOT EXISTS audit_runs_audit_id_idx ON audit_runs(audit_id);
CREATE INDEX IF NOT EXISTS audit_runs_status_idx ON audit_runs(status);
"""


def default_home() -> Path:
    """Return the audit cache directory, creating it if needed."""
    raw = os.environ.get("VLABS_AUDIT_HOME") or "~/.vlabs-audit"
    return Path(raw).expanduser()


def default_db_path() -> Path:
    return default_home() / "audits.db"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AuditRecord:
    id: str
    model: str
    config: dict[str, Any]
    started_at: str
    finished_at: str | None


@dataclass(frozen=True)
class AuditRunRecord:
    id: int
    audit_id: str
    env: str
    episode_idx: int
    seed: int
    status: str
    jsonl_path: str | None
    reward: float | None
    error: str | None
    started_at: str | None
    finished_at: str | None


class AuditStore:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions manually
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    # ── audits ────────────────────────────────────────────────────

    def create_audit(self, model: str, config: dict[str, Any]) -> str:
        audit_id = f"aud_{uuid.uuid4().hex}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO audits(id, model, config_json, started_at) "
                "VALUES (?, ?, ?, ?)",
                (audit_id, model, json.dumps(config, default=str), _now_iso()),
            )
        return audit_id

    def get_audit(self, audit_id: str) -> AuditRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM audits WHERE id = ?", (audit_id,)
            ).fetchone()
        if row is None:
            return None
        return AuditRecord(
            id=row["id"],
            model=row["model"],
            config=json.loads(row["config_json"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def list_audits(self) -> list[AuditRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audits ORDER BY started_at DESC"
            ).fetchall()
        return [
            AuditRecord(
                id=r["id"],
                model=r["model"],
                config=json.loads(r["config_json"]),
                started_at=r["started_at"],
                finished_at=r["finished_at"],
            )
            for r in rows
        ]

    def finish_audit(self, audit_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE audits SET finished_at = ? WHERE id = ?",
                (_now_iso(), audit_id),
            )

    # ── audit_runs ────────────────────────────────────────────────

    def schedule_episodes(
        self,
        audit_id: str,
        env: str,
        episodes: int,
        seed_start: int,
    ) -> int:
        """Idempotently insert ``episodes`` pending rows for (audit_id, env).

        Re-runs are safe — the ``UNIQUE(audit_id, env, episode_idx)``
        constraint plus ``INSERT OR IGNORE`` mean a re-call inserts
        nothing new and just returns the existing pending count.
        """
        with self._lock:
            cur = self._conn.cursor()
            inserted = 0
            for i in range(episodes):
                cur.execute(
                    "INSERT OR IGNORE INTO audit_runs"
                    "(audit_id, env, episode_idx, seed) "
                    "VALUES (?, ?, ?, ?)",
                    (audit_id, env, i, seed_start + i),
                )
                inserted += cur.rowcount
        return inserted

    def list_pending(self, audit_id: str) -> list[AuditRunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_runs "
                "WHERE audit_id = ? AND status = 'pending' "
                "ORDER BY id",
                (audit_id,),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def list_runs(self, audit_id: str) -> list[AuditRunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_runs WHERE audit_id = ? ORDER BY id",
                (audit_id,),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def reset_stale_running(self, audit_id: str) -> int:
        """Crashed prior run? Move ``running`` rows back to ``pending``."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE audit_runs SET status = 'pending', started_at = NULL "
                "WHERE audit_id = ? AND status = 'running'",
                (audit_id,),
            )
            return cur.rowcount

    def mark_running(self, run_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE audit_runs SET status = 'running', started_at = ? "
                "WHERE id = ?",
                (_now_iso(), run_id),
            )

    def complete_episode(
        self,
        run_id: int,
        reward: float,
        jsonl_path: Path,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE audit_runs SET status = 'success', reward = ?, "
                "jsonl_path = ?, finished_at = ? WHERE id = ?",
                (float(reward), str(jsonl_path), _now_iso(), run_id),
            )

    def fail_episode(self, run_id: int, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE audit_runs SET status = 'failed', error = ?, "
                "finished_at = ? WHERE id = ?",
                (error[:1000], _now_iso(), run_id),
            )

    def counts_by_status(self, audit_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM audit_runs "
                "WHERE audit_id = ? GROUP BY status",
                (audit_id,),
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    # ── lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> AuditRunRecord:
        return AuditRunRecord(
            id=int(row["id"]),
            audit_id=row["audit_id"],
            env=row["env"],
            episode_idx=int(row["episode_idx"]),
            seed=int(row["seed"]),
            status=row["status"],
            jsonl_path=row["jsonl_path"],
            reward=float(row["reward"]) if row["reward"] is not None else None,
            error=row["error"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


__all__ = [
    "AuditRecord",
    "AuditRunRecord",
    "AuditStore",
    "default_db_path",
    "default_home",
]
