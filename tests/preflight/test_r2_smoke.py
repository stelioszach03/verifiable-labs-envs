"""Tests for scripts/preflight/r2_smoke.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "r2_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("r2_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r2():
    return _load()


def test_run_smoke_writes_json(r2, tmp_path) -> None:
    out = tmp_path / "r2.json"
    report = r2.run_smoke(out=out)
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_run_smoke_returns_ok(r2, tmp_path) -> None:
    report = r2.run_smoke(out=tmp_path / "r2.json")
    assert report["ok"] is True


def test_run_smoke_reports_real_r2_unconfigured(r2, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    report = r2.run_smoke(out=tmp_path / "r2.json")
    assert report["is_real_r2_configured"] is False


def test_reward_train_uploader_sha_match(r2, tmp_path) -> None:
    report = r2.run_smoke(out=tmp_path / "r2.json")
    rt = report["reward_train_uploader"]
    assert rt["sha_match"] is True
    assert rt["sha256"] == report["local_sha256"]


def test_process_reward_uploader_sha_match(r2, tmp_path) -> None:
    report = r2.run_smoke(out=tmp_path / "r2.json")
    prm = report["process_reward_uploader"]
    assert prm["sha_match"] is True
    assert prm["sha256"] == report["local_sha256"]


def test_both_uploaders_emit_uri(r2, tmp_path) -> None:
    report = r2.run_smoke(out=tmp_path / "r2.json")
    assert "r2://" in report["reward_train_uploader"]["uri"]
    assert "r2://" in report["process_reward_uploader"]["uri"]


def test_uploaders_record_latencies(r2, tmp_path) -> None:
    report = r2.run_smoke(out=tmp_path / "r2.json")
    assert (
        report["reward_train_uploader"]["latency_seconds"] >= 0.0
    )
    assert (
        report["process_reward_uploader"]["latency_seconds"] >= 0.0
    )


def test_main_cli_exits_zero(r2, tmp_path) -> None:
    rc = r2.main(["--out", str(tmp_path / "main.json"), "--quiet"])
    assert rc == 0
    assert (tmp_path / "main.json").is_file()
