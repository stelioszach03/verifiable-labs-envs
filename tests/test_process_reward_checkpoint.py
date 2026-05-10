"""Tests for ``verifiable_labs_envs.process_reward.checkpoint``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_labs_envs.process_reward.checkpoint import (
    LORA_WEIGHTS_FILENAME,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    PrmCheckpointManifest,
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


def _build_manifest(quantile: float | None = 0.087) -> PrmCheckpointManifest:
    return PrmCheckpointManifest(
        model_id="vlabs-prm-distilled-qwen-1-5b-v0.1.0",
        version="0.1.0",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        step_granularity="per_step",
        base_rm_id="vlabs-reward-distilled-qwen-1-5b-v0.1.0",
        lora_config={"r": 16, "alpha": 32},
        training_config={"lr": 1e-4, "epochs": 3},
        multi_task={"enable": True, "per_step_weight": 0.7, "outcome_weight": 0.3},
        metrics={"processbench_overall": 0.62},
        checkpoint_files=(LORA_WEIGHTS_FILENAME,),
        step_conformal_quantiles={"range(0, 1)": 0.05, "range(1, 3)": 0.07},
        aggregate_conformal_quantile=quantile,
    )


# ── model_id_for ───────────────────────────────────────────────────


def test_model_id_for_locked_shape() -> None:
    """Plan §5 D12-B: vlabs-prm-{family}-v{semver}."""
    assert (
        model_id_for("distilled-qwen-1-5b", "0.1.0")
        == "vlabs-prm-distilled-qwen-1-5b-v0.1.0"
    )


def test_model_id_for_rejects_empty() -> None:
    with pytest.raises(ValueError, match="family"):
        model_id_for("", "0.1.0")
    with pytest.raises(ValueError, match="version"):
        model_id_for("x", "")


# ── manifest round trip ────────────────────────────────────────────


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _build_manifest()
    target = write_manifest(tmp_path, manifest)
    assert target == tmp_path / MANIFEST_FILENAME
    restored = read_manifest(target)
    assert restored.model_id == manifest.model_id
    assert restored.base_rm_id == manifest.base_rm_id
    assert restored.step_granularity == "per_step"
    assert restored.step_conformal_quantiles == manifest.step_conformal_quantiles
    assert restored.multi_task == manifest.multi_task


def test_read_manifest_accepts_directory(tmp_path: Path) -> None:
    manifest = _build_manifest()
    write_manifest(tmp_path, manifest)
    restored = read_manifest(tmp_path)
    assert restored.model_id == manifest.model_id


def test_manifest_fingerprint_stable() -> None:
    a = _build_manifest()
    b = _build_manifest()
    # Same inputs → same fingerprint despite distinct timestamps.
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 64


def test_manifest_to_dict_includes_step_granularity() -> None:
    d = _build_manifest().to_dict()
    assert d["step_granularity"] == "per_step"
    assert d["schema_version"] == SCHEMA_VERSION


def test_manifest_independent_serving_has_no_base_rm_id() -> None:
    manifest = PrmCheckpointManifest(
        model_id="vlabs-prm-foo-v0.0.1",
        version="0.0.1",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        step_granularity="per_step",
        base_rm_id=None,
        lora_config={},
        training_config={},
        multi_task={},
        metrics={},
        checkpoint_files=(),
    )
    assert manifest.base_rm_id is None


# ── training config + R2 ───────────────────────────────────────────


def test_write_training_config_persists(tmp_path: Path) -> None:
    cfg = {"lr": 1e-4, "epochs": 3, "multi_task": True}
    target = write_training_config(tmp_path, cfg)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == cfg


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
    uploaded = upload_checkpoint(
        tmp_path, manifest, uploader=fake_r2_uploader(fake_root)
    )
    assert LORA_WEIGHTS_FILENAME in uploaded
    assert MANIFEST_FILENAME in uploaded


def test_upload_checkpoint_rejects_missing_dir(tmp_path: Path) -> None:
    manifest = _build_manifest()
    with pytest.raises(FileNotFoundError):
        upload_checkpoint(tmp_path / "missing", manifest)


# ── list_local_checkpoints ─────────────────────────────────────────


def test_list_local_checkpoints_recursive(tmp_path: Path) -> None:
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
    assert row["base_rm_id"] == manifest.base_rm_id
    assert row["step_granularity"] == "per_step"
    assert row["step_buckets"] == sorted(manifest.step_conformal_quantiles)


def test_is_real_r2_configured_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    assert is_real_r2_configured() is False


def test_is_real_r2_configured_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "y")
    assert is_real_r2_configured() is True
