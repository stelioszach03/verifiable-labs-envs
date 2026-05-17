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


def test_dry_run_carries_29f_flags(tmp_path: Path) -> None:
    """Phase 29.F: vLLM + env + max-steps flags surface through to the
    config and the resolved training_args dict."""
    out_dir = tmp_path / "exp_29f"
    result = runner.invoke(
        app,
        [
            "dry-run",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(out_dir),
            "--max-steps", "10",
            "--num-generations", "4",
            "--env-id", "math-algebra",
            "--vllm-gpu-memory-utilization", "0.4",
            "--vllm-max-model-length", "2048",
            "--beta", "0.05",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["config"]["max_steps"] == 10
    assert payload["config"]["num_generations"] == 4
    assert payload["config"]["env_id"] == "math-algebra"
    assert payload["config"]["vllm_gpu_memory_utilization"] == 0.4
    assert payload["config"]["vllm_max_model_length"] == 2048
    assert payload["config"]["beta"] == 0.05
    # training_args dict mirrors the rename:
    args = payload["training_args"]
    assert args["max_steps"] == 10
    assert args["num_generations"] == 4
    assert args["beta"] == 0.05
    assert args["vllm_max_model_length"] == 2048


def test_train_command_aborts_cleanly_on_unusable_invocation(
    tmp_path: Path,
) -> None:
    """In Phase 29.F the ``train`` command exits with a non-zero
    code (and an ABORT message on stderr) when either dependencies
    are missing OR the dataset doesn't exist. We don't assert which
    of the two — both are valid clean aborts. The test ensures the
    CLI never accidentally proceeds to a doomed real run."""
    result = runner.invoke(
        app,
        [
            "train",
            "--dataset", str(tmp_path / "fake.jsonl"),
            "--output-dir", str(tmp_path / "exp"),
        ],
    )
    # 2 = missing deps, 4 = missing dataset. Either is acceptable.
    assert result.exit_code in (2, 4), result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "ABORT" in combined


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
