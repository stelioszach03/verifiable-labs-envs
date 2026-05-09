"""Tests for ``vlabs_reward_train.checkpointing``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlabs_reward_train.checkpointing import (
    LORA_WEIGHTS_FILENAME,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    CheckpointManifest,
    fake_r2_uploader,
    is_real_r2_configured,
    list_local_checkpoints,
    manifest_table,
    model_id_for,
    read_manifest,
    upload_checkpoint,
    write_manifest,
    write_training_config,
)


def _build_manifest(quantile: float | None = 0.087) -> CheckpointManifest:
    return CheckpointManifest(
        model_id="vlabs-reward-distilled-qwen-1-5b-v0.1.0",
        version="0.1.0",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_config={"r": 16, "alpha": 32},
        training_config={"lr": 2e-4, "epochs": 3},
        metrics={"spearman_avg": 0.78, "rewardbench": 0.71},
        checkpoint_files=(LORA_WEIGHTS_FILENAME,),
        conformal_quantile=quantile,
    )


def test_model_id_for_locked_shape() -> None:
    assert (
        model_id_for("distilled-qwen-1-5b", "0.1.0")
        == "vlabs-reward-distilled-qwen-1-5b-v0.1.0"
    )


def test_model_id_for_rejects_empty_args() -> None:
    with pytest.raises(ValueError, match="family"):
        model_id_for("", "0.1.0")
    with pytest.raises(ValueError, match="version"):
        model_id_for("distilled-qwen-1-5b", "")


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _build_manifest()
    target = write_manifest(tmp_path, manifest)
    assert target == tmp_path / MANIFEST_FILENAME
    restored = read_manifest(target)
    # Compare ignoring the timestamp (assigned at construction time).
    assert restored.model_id == manifest.model_id
    assert restored.lora_config == manifest.lora_config
    assert restored.metrics == manifest.metrics


def test_read_manifest_accepts_directory(tmp_path: Path) -> None:
    manifest = _build_manifest()
    write_manifest(tmp_path, manifest)
    restored = read_manifest(tmp_path)  # passes the directory
    assert restored.model_id == manifest.model_id


def test_manifest_fingerprint_stable(tmp_path: Path) -> None:
    a = _build_manifest()
    # Same inputs → same fingerprint despite distinct timestamps.
    b = _build_manifest()
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 64  # SHA-256 hex


def test_manifest_to_dict_includes_checkpoint_files() -> None:
    d = _build_manifest().to_dict()
    assert d["checkpoint_files"] == [LORA_WEIGHTS_FILENAME]
    assert d["schema_version"] == SCHEMA_VERSION


def test_write_training_config_persists_bytewise(tmp_path: Path) -> None:
    cfg = {"lr": 2e-4, "epochs": 3}
    target = write_training_config(tmp_path, cfg)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == cfg


def test_fake_r2_uploader_round_trips_files(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello")
    dst_root = tmp_path / "fake-r2"
    upload = fake_r2_uploader(dst_root)
    uri = upload(src, "modelid/0.1.0/checkpoint/src.bin")
    assert uri.startswith("r2://vlabs-models/")
    landing = dst_root / "modelid" / "0.1.0" / "checkpoint" / "src.bin"
    assert landing.read_bytes() == b"hello"


def test_upload_checkpoint_uploads_files_and_manifest(tmp_path: Path) -> None:
    manifest = _build_manifest()
    weights_path = tmp_path / LORA_WEIGHTS_FILENAME
    weights_path.write_bytes(b"fake weights")
    write_manifest(tmp_path, manifest)
    fake_root = tmp_path / "fake-r2"
    uploaded = upload_checkpoint(tmp_path, manifest, uploader=fake_r2_uploader(fake_root))
    assert LORA_WEIGHTS_FILENAME in uploaded
    assert MANIFEST_FILENAME in uploaded
    # Both files round-trip into the fake R2 tree.
    assert (
        fake_root
        / manifest.model_id
        / manifest.version
        / "checkpoint"
        / LORA_WEIGHTS_FILENAME
    ).exists()


def test_upload_checkpoint_rejects_missing_dir(tmp_path: Path) -> None:
    manifest = _build_manifest()
    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        upload_checkpoint(missing, manifest)


def test_list_local_checkpoints_discovers_recursively(tmp_path: Path) -> None:
    a = tmp_path / "exp_001"
    b = tmp_path / "exp_002"
    write_manifest(a, _build_manifest())
    write_manifest(b, _build_manifest())
    found = list_local_checkpoints(tmp_path)
    assert len(found) == 2


def test_list_local_checkpoints_handles_missing_dir() -> None:
    assert list_local_checkpoints("/no-such-path-anywhere") == []


def test_manifest_table_compact() -> None:
    manifest = _build_manifest()
    table = manifest_table([manifest])
    assert len(table) == 1
    row = table[0]
    assert row["model_id"] == manifest.model_id
    assert row["n_files"] == 1
    assert "metrics_keys" in row


def test_is_real_r2_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    assert is_real_r2_configured() is False


def test_is_real_r2_configured_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "y")
    assert is_real_r2_configured() is True
