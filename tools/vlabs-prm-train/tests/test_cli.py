"""Tests for ``vlabs_prm_train.cli``."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vlabs_prm_train import __version__
from vlabs_prm_train.cli import app

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    for cmd in ("train", "dry-run", "dependencies", "version", "checkpoints"):
        assert cmd in out


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_dependencies_emits_json() -> None:
    result = runner.invoke(app, ["dependencies"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "available" in payload
    assert "missing" in payload
    assert "wandb_installed" in payload


def test_dry_run_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp_001"
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(out_dir),
            "--epochs", "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["config"]["epochs"] == 2
    assert "multi_task" in payload
    assert "training_args" in payload


def test_dry_run_independent_serving_default(tmp_path: Path) -> None:
    """Without --base-rm-checkpoint, shared_backbone_ready is False."""
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["shared_backbone_ready"] is False
    assert payload["config"]["base_rm_checkpoint"] is None


def test_dry_run_shared_backbone_when_checkpoint_passed(tmp_path: Path) -> None:
    rm_ckpt = tmp_path / "rm-ckpt"
    rm_ckpt.mkdir()
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "out"),
            "--base-rm-checkpoint", str(rm_ckpt),
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["shared_backbone_ready"] is True
    assert str(rm_ckpt) in payload["config"]["base_rm_checkpoint"]


def test_dry_run_writes_run_card(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp_002"
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(out_dir),
            "--write-run-card",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert (out_dir / "run_card.json").exists()
    assert payload["run_card_path"]


def test_dry_run_multi_task_toggle(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "out"),
            "--multi-task",
            "--multi-task-outcome-weight", "0.4",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["multi_task"]["enable"] is True
    assert payload["multi_task"]["outcome_weight"] == 0.4


def test_train_aborts_on_missing_deps(tmp_path: Path) -> None:
    """In CI without GPU deps, exit 2 with ABORT message."""
    result = runner.invoke(
        app,
        [
            "train",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "exp"),
        ],
    )
    assert result.exit_code in (2, 3)
    if result.exit_code == 2:
        assert "ABORT" in (result.stdout + result.stderr)


def test_checkpoints_empty_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["checkpoints", "--parent", str(tmp_path)])
    assert result.exit_code == 0
    table = json.loads(result.stdout)
    assert table == []


def test_checkpoints_finds_local_manifest(tmp_path: Path) -> None:
    from verifiable_labs_envs.process_reward.checkpoint import (
        PrmCheckpointManifest,
        write_manifest,
    )

    manifest = PrmCheckpointManifest(
        model_id="vlabs-prm-distilled-qwen-1-5b-v0.1.0",
        version="0.1.0",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        step_granularity="per_step",
        base_rm_id=None,
        lora_config={},
        training_config={},
        multi_task={},
        metrics={},
        checkpoint_files=("adapter_model.safetensors",),
    )
    write_manifest(tmp_path / "exp_001", manifest)
    result = runner.invoke(app, ["checkpoints", "--parent", str(tmp_path)])
    assert result.exit_code == 0
    table = json.loads(result.stdout)
    assert len(table) == 1
    assert table[0]["model_id"] == manifest.model_id


def test_help_mentions_phase_30() -> None:
    result = runner.invoke(app, ["--help"])
    assert "Phase 30" in result.stdout
