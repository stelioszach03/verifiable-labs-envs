"""Tests for scripts/preflight/checkpoint_roundtrip.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "preflight" / "checkpoint_roundtrip.py"


def _load():
    spec = importlib.util.spec_from_file_location("checkpoint_roundtrip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cr():
    return _load()


# ── deterministic payload + hashing ───────────────────────────────


def test_fake_payload_is_deterministic(cr) -> None:
    a = cr.fake_payload(seed=42)
    b = cr.fake_payload(seed=42)
    assert a == b
    assert len(a) == cr.DEFAULT_PAYLOAD_BYTES


def test_fake_payload_seed_changes_bytes(cr) -> None:
    assert cr.fake_payload(seed=0) != cr.fake_payload(seed=1)


def test_sha256_is_64_hex_chars(cr) -> None:
    out = cr.sha256(b"hello")
    assert len(out) == 64
    assert all(c in "0123456789abcdef" for c in out)


def test_fake_manifest_carries_payload_sha(cr) -> None:
    sha = cr.sha256(b"payload")
    m = cr.fake_manifest(sha)
    assert m["payload_sha256"] == sha
    assert m["smoke"] is True
    assert m["model_id"].startswith("vlabs-reward-distilled")


# ── round-trip in fake-HF mode ────────────────────────────────────


def test_run_smoke_writes_json(cr, tmp_path) -> None:
    out = tmp_path / "ckpt.json"
    report = cr.run_smoke(out=out, fake_root=tmp_path / "fake-hf")
    assert out.is_file()
    assert json.loads(out.read_text()) == report


def test_run_smoke_returns_ok_in_fake_mode(cr, tmp_path) -> None:
    report = cr.run_smoke(
        out=tmp_path / "ckpt.json",
        fake_root=tmp_path / "fake-hf",
        use_local_fake_hf=True,
    )
    assert report["ok"] is True
    assert report["mode"] == "local-fake-hf"


def test_round_trip_preserves_bytes_and_hash(cr, tmp_path) -> None:
    report = cr.run_smoke(
        out=tmp_path / "ckpt.json",
        fake_root=tmp_path / "fake-hf",
        use_local_fake_hf=True,
    )
    assert report["bytes_match"] is True
    assert report["sha_match"] is True
    assert report["payload_sha256_upload"] == report["payload_sha256_download"]


def test_round_trip_preserves_manifest(cr, tmp_path) -> None:
    report = cr.run_smoke(
        out=tmp_path / "ckpt.json",
        fake_root=tmp_path / "fake-hf",
        use_local_fake_hf=True,
    )
    assert report["manifest_match"] is True


def test_round_trip_records_latencies(cr, tmp_path) -> None:
    report = cr.run_smoke(
        out=tmp_path / "ckpt.json",
        fake_root=tmp_path / "fake-hf",
        use_local_fake_hf=True,
    )
    assert report["upload_latency_seconds"] >= 0.0
    assert report["download_latency_seconds"] >= 0.0


def test_round_trip_writes_files_under_fake_root(cr, tmp_path) -> None:
    fake_root = tmp_path / "fake-hf"
    cr.run_smoke(
        out=tmp_path / "ckpt.json",
        fake_root=fake_root,
        repo_id="acme/preflight-smoke",
        use_local_fake_hf=True,
    )
    repo_dir = fake_root / "acme__preflight-smoke"
    assert (repo_dir / "adapter_model.safetensors").is_file()
    assert (repo_dir / "manifest.json").is_file()


def test_run_smoke_overwrites_existing_repo_dir(cr, tmp_path) -> None:
    """A second invocation must not fail because of stale files."""
    fake_root = tmp_path / "fake-hf"
    cr.run_smoke(
        out=tmp_path / "a.json",
        fake_root=fake_root,
        seed=0,
        use_local_fake_hf=True,
    )
    report2 = cr.run_smoke(
        out=tmp_path / "b.json",
        fake_root=fake_root,
        seed=1,
        use_local_fake_hf=True,
    )
    assert report2["ok"] is True


def test_run_smoke_picks_fake_mode_when_no_hf_token(
    cr, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LOCAL_FAKE_HF", raising=False)
    report = cr.run_smoke(
        out=tmp_path / "ckpt.json", fake_root=tmp_path / "fake-hf"
    )
    assert report["mode"] == "local-fake-hf"


def test_main_cli_exit_zero_on_success(cr, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = cr.main(
        [
            "--out",
            str(tmp_path / "main.json"),
            "--fake-root",
            str(tmp_path / "fake-hf"),
            "--quiet",
        ]
    )
    assert rc == 0
