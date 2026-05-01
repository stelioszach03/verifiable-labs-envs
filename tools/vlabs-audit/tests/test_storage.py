"""Unit tests for ``vlabs_audit.storage`` — SQLite CRUD + state machine."""
from __future__ import annotations

from pathlib import Path

import pytest

from vlabs_audit.storage import (
    AuditStore,
    default_db_path,
    default_home,
)


def test_create_get_finish_audit(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("claude-haiku-4.5", {"k": "v", "envs": ["a"]})
        assert aid.startswith("aud_")

        rec = store.get_audit(aid)
        assert rec is not None
        assert rec.id == aid
        assert rec.model == "claude-haiku-4.5"
        assert rec.config == {"k": "v", "envs": ["a"]}
        assert rec.finished_at is None

        store.finish_audit(aid)
        rec2 = store.get_audit(aid)
        assert rec2 is not None
        assert rec2.finished_at is not None

        listed = store.list_audits()
        assert [r.id for r in listed] == [aid]

        assert store.get_audit("aud_nope") is None


def test_schedule_episodes_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {})
        n1 = store.schedule_episodes(aid, "env-a", episodes=5, seed_start=100)
        n2 = store.schedule_episodes(aid, "env-a", episodes=5, seed_start=100)
        assert n1 == 5
        assert n2 == 0

        runs = store.list_runs(aid)
        assert len(runs) == 5
        assert sorted(r.seed for r in runs) == list(range(100, 105))
        assert {r.episode_idx for r in runs} == set(range(5))
        assert all(r.status == "pending" for r in runs)

        # Second env on same audit gets its own rows.
        n3 = store.schedule_episodes(aid, "env-b", episodes=2, seed_start=0)
        assert n3 == 2
        assert len(store.list_runs(aid)) == 7


def test_state_machine_success_and_failure(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {})
        store.schedule_episodes(aid, "env-a", episodes=2, seed_start=0)
        pending = store.list_pending(aid)
        assert len(pending) == 2

        store.mark_running(pending[0].id)
        store.complete_episode(
            pending[0].id, reward=0.85, jsonl_path=tmp_path / "ok.jsonl"
        )

        store.mark_running(pending[1].id)
        store.fail_episode(pending[1].id, "RuntimeError: boom")

        counts = store.counts_by_status(aid)
        assert counts == {"success": 1, "failed": 1}

        runs = {r.id: r for r in store.list_runs(aid)}
        assert runs[pending[0].id].status == "success"
        assert runs[pending[0].id].reward == 0.85
        assert runs[pending[0].id].jsonl_path == str(tmp_path / "ok.jsonl")
        assert runs[pending[0].id].finished_at is not None

        assert runs[pending[1].id].status == "failed"
        assert runs[pending[1].id].error is not None
        assert "RuntimeError" in runs[pending[1].id].error

        # No pending rows left to drain.
        assert store.list_pending(aid) == []


def test_reset_stale_running_recovers_pending(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    with AuditStore(db_path=db) as store:
        aid = store.create_audit("m", {})
        store.schedule_episodes(aid, "env-a", episodes=3, seed_start=0)
        pending = store.list_pending(aid)
        for r in pending:
            store.mark_running(r.id)
        assert store.counts_by_status(aid) == {"running": 3}

        n = store.reset_stale_running(aid)
        assert n == 3
        assert store.counts_by_status(aid) == {"pending": 3}
        # And started_at is cleared so the row looks freshly pending.
        for r in store.list_pending(aid):
            assert r.started_at is None

        # Idempotent: a second call resets nothing because no rows are 'running'.
        assert store.reset_stale_running(aid) == 0


def test_default_home_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLABS_AUDIT_HOME", str(tmp_path / "alt"))
    assert default_home() == tmp_path / "alt"
    assert default_db_path() == tmp_path / "alt" / "audits.db"

    monkeypatch.delenv("VLABS_AUDIT_HOME", raising=False)
    assert default_home() == Path("~/.vlabs-audit").expanduser()
