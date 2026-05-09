"""Tests for the ``vlabs-reward-train`` CLI."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vlabs_reward_train import __version__
from vlabs_reward_train.cli import app

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


def test_dependencies_command_emits_json() -> None:
    result = runner.invoke(app, ["dependencies"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "available" in payload
    assert "missing" in payload
    assert "wandb_installed" in payload


def test_dry_run_smoke(tmp_path: Path) -> None:
    """Dry-run prints config + dep status and exits 0 without training."""
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
    assert "lora_summary" in payload
    assert "training_args" in payload


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
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "run_card_path" in payload
    assert (out_dir / "run_card.json").exists()


def test_train_command_aborts_on_missing_deps(tmp_path: Path) -> None:
    """In CI without GPU deps, the train command exits 2 with an
    "ABORT: missing dependencies" message."""
    result = runner.invoke(
        app,
        [
            "train",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "exp"),
        ],
    )
    # Exit code 2 = missing dependencies; 3 = GPU path not implemented.
    # In CI without trl/peft we get exit 2.
    assert result.exit_code in (2, 3)
    if result.exit_code == 2:
        assert "ABORT" in (result.stdout + result.stderr)


def test_checkpoints_command_empty_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["checkpoints", "--parent", str(tmp_path)])
    assert result.exit_code == 0
    table = json.loads(result.stdout)
    assert table == []


def test_checkpoints_command_finds_local_manifest(tmp_path: Path) -> None:
    from vlabs_reward_train.checkpointing import (
        CheckpointManifest,
        write_manifest,
    )

    manifest = CheckpointManifest(
        model_id="vlabs-reward-distilled-qwen-1-5b-v0.1.0",
        version="0.1.0",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_config={},
        training_config={},
        metrics={},
        checkpoint_files=("adapter_model.safetensors",),
    )
    write_manifest(tmp_path / "exp_001", manifest)
    result = runner.invoke(app, ["checkpoints", "--parent", str(tmp_path)])
    assert result.exit_code == 0
    table = json.loads(result.stdout)
    assert len(table) == 1
    assert table[0]["model_id"] == manifest.model_id


def test_dry_run_help_text_mentions_phase() -> None:
    result = runner.invoke(app, ["--help"])
    assert "Phase 29" in result.stdout
