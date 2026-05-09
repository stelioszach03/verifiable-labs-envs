"""Tests for ``vlabs_reward_train.trainer``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlabs_reward_train.trainer import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LR,
    REQUIRED_DEPS,
    DependencyStatus,
    GpuPathNotImplemented,
    TrainingConfig,
    build_grpo_trainer,
    build_training_args,
    validate_dependencies,
    write_run_card,
)


def test_default_base_model_locked_per_plan() -> None:
    """Plan §5 D2-A: Qwen2.5-1.5B-Instruct."""
    assert DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert pytest.approx(2e-4) == DEFAULT_LR


def test_required_deps_includes_torch_trl_peft() -> None:
    assert "torch" in REQUIRED_DEPS
    assert "trl" in REQUIRED_DEPS
    assert "peft" in REQUIRED_DEPS
    assert "transformers" in REQUIRED_DEPS


def test_training_config_round_trip() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    payload = cfg.to_dict()
    restored = TrainingConfig.from_dict(payload)
    assert restored == cfg


def test_training_config_with_overrides_creates_new() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    new = cfg.with_overrides(lr=1e-5, epochs=10)
    assert new.lr == pytest.approx(1e-5)
    assert new.epochs == 10
    assert cfg.lr != new.lr  # original untouched


def test_training_config_lora_spec_pulls_through() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", lora_r=8, lora_alpha=16)
    spec = cfg.lora_spec
    assert spec.r == 8
    assert spec.alpha == 16


def test_validate_dependencies_returns_status() -> None:
    status = validate_dependencies()
    assert isinstance(status, DependencyStatus)
    # In CI without GPU deps, missing is non-empty.
    assert isinstance(status.missing, tuple)
    assert isinstance(status.available, tuple)
    assert status.is_satisfied == (not status.missing)


def test_validate_dependencies_with_minimal_required() -> None:
    """Probe with a known-installed dep so we hit the available branch."""
    status = validate_dependencies(required=("json",))
    assert status.is_satisfied
    assert "json" in status.available


def test_validate_dependencies_status_to_dict() -> None:
    status = DependencyStatus(available=("torch",), missing=("trl",))
    assert status.to_dict() == {
        "available": ["torch"],
        "missing": ["trl"],
        "is_satisfied": False,
    }


def test_build_training_args_basic() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    args = build_training_args(cfg)
    assert args["learning_rate"] == pytest.approx(DEFAULT_LR)
    assert args["num_train_epochs"] == cfg.epochs
    assert args["per_device_train_batch_size"] == cfg.batch_size
    assert args["bf16"] is True
    assert args["report_to"] == ["wandb"]


def test_build_training_args_disabled_wandb_drops_report_to() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", wandb_mode="disabled")
    args = build_training_args(cfg)
    assert args["report_to"] == []


def test_build_training_args_rejects_missing_dataset() -> None:
    cfg = TrainingConfig(dataset_path="")
    with pytest.raises(ValueError, match="dataset_path"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_lr() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", lr=1.5)
    with pytest.raises(ValueError, match="lr"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_epochs() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl", epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        build_training_args(cfg)


def test_build_grpo_trainer_raises_in_29c() -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    with pytest.raises(GpuPathNotImplemented, match="29.F"):
        build_grpo_trainer(cfg)


def test_write_run_card_persists_payload(tmp_path: Path) -> None:
    cfg = TrainingConfig(dataset_path="/tmp/x.jsonl")
    status = DependencyStatus(available=("torch",), missing=("trl",))
    target = write_run_card(tmp_path, cfg, status)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["config"]["base_model"] == DEFAULT_BASE_MODEL
    assert payload["dependencies"]["is_satisfied"] is False
    assert payload["schema_version"] == "v0.1.0"
