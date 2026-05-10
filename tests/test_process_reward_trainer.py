"""Tests for ``verifiable_labs_envs.process_reward.trainer``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_labs_envs.process_reward.dataset import (
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
)
from verifiable_labs_envs.process_reward.trainer import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LR,
    DEFAULT_MAX_TRACE_LENGTH,
    REQUIRED_DEPS,
    DependencyStatus,
    GpuPathNotImplemented,
    PrmTrainingConfig,
    build_prm_trainer,
    build_training_args,
    per_step_outcome_tensor,
    per_step_target_tensor,
    validate_dependencies,
    write_run_card,
)


def _trace(env_rewards: tuple[float, ...]) -> ProcessRewardTraceRow:
    n = len(env_rewards)
    return ProcessRewardTraceRow(
        row_id="prw_x",
        env_id="math-algebra",
        prompt="p",
        steps=tuple(f"step-{i}" for i in range(n)),
        step_rewards=env_rewards,
        step_components=tuple(None for _ in range(n)),
        step_conformal_intervals=tuple(None for _ in range(n)),
        step_frontier_judgments=tuple(None for _ in range(n)),
        step_frontier_rationales=tuple(None for _ in range(n)),
        step_consensus_rewards=tuple(env_rewards),
        step_disagreements=tuple(None for _ in range(n)),
        aggregate_reward=sum(env_rewards) / max(1, n),
        aggregate_conformal_interval=None,
        decomposition="text_progress",
        segmentation_strategy="explicit_step_marker",
        segmentation_confidence=0.95,
        truncated=False,
        source="env",
        metadata={"schema_version": SCHEMA_VERSION},
    )


# ── locked defaults ─────────────────────────────────────────────────


def test_default_base_model_locked() -> None:
    """Plan §5 D3-A: Qwen2.5-1.5B-Instruct."""
    assert DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_default_lr_lower_than_phase29() -> None:
    """Plan §8: PRM uses gentler LR (1e-4 vs Phase 29's 2e-4)."""
    assert pytest.approx(1e-4) == DEFAULT_LR


def test_default_max_trace_length_locked() -> None:
    """Plan §8: traces are longer than completions, max=4096."""
    assert DEFAULT_MAX_TRACE_LENGTH == 4096


def test_required_deps_includes_torch_trl_peft() -> None:
    for dep in ("torch", "transformers", "peft", "trl", "accelerate"):
        assert dep in REQUIRED_DEPS


# ── PrmTrainingConfig round-trip ───────────────────────────────────


def test_config_round_trip() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl", multi_task=True)
    payload = cfg.to_dict()
    restored = PrmTrainingConfig.from_dict(payload)
    assert restored == cfg


def test_config_with_overrides() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl")
    new = cfg.with_overrides(lr=5e-5, multi_task=True)
    assert new.lr == pytest.approx(5e-5)
    assert new.multi_task is True
    assert cfg.lr != new.lr


def test_config_shared_backbone_predicate() -> None:
    cfg_indep = PrmTrainingConfig(dataset_path="/tmp/x.jsonl")
    cfg_shared = PrmTrainingConfig(
        dataset_path="/tmp/x.jsonl",
        base_rm_checkpoint="runs/reward-train/exp_004/",
    )
    assert cfg_indep.shared_backbone is False
    assert cfg_shared.shared_backbone is True


def test_per_step_loss_weight_complement() -> None:
    cfg = PrmTrainingConfig(
        dataset_path="/tmp/x.jsonl", multi_task_outcome_weight=0.4
    )
    assert cfg.per_step_loss_weight == pytest.approx(0.6)


# ── validate_dependencies ──────────────────────────────────────────


def test_validate_dependencies_returns_status() -> None:
    status = validate_dependencies()
    assert isinstance(status, DependencyStatus)
    assert isinstance(status.missing, tuple)
    assert isinstance(status.available, tuple)
    assert status.is_satisfied == (not status.missing)


def test_validate_dependencies_with_minimal_required() -> None:
    """Probe with a known-installed dep so we hit the available branch."""
    status = validate_dependencies(required=("json",))
    assert status.is_satisfied
    assert "json" in status.available


def test_dependency_status_to_dict() -> None:
    status = DependencyStatus(available=("torch",), missing=("trl",))
    payload = status.to_dict()
    assert payload == {
        "available": ["torch"],
        "missing": ["trl"],
        "is_satisfied": False,
    }


# ── build_training_args ────────────────────────────────────────────


def test_build_training_args_basic() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl")
    args = build_training_args(cfg)
    assert args["learning_rate"] == pytest.approx(DEFAULT_LR)
    assert args["num_train_epochs"] == cfg.epochs
    assert args["bf16"] is True
    assert args["report_to"] == ["wandb"]


def test_build_training_args_disabled_wandb_drops_report_to() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl", wandb_mode="disabled")
    args = build_training_args(cfg)
    assert args["report_to"] == []


def test_build_training_args_rejects_missing_dataset() -> None:
    with pytest.raises(ValueError, match="dataset_path"):
        build_training_args(PrmTrainingConfig(dataset_path=""))


def test_build_training_args_rejects_invalid_lr() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl", lr=2.0)
    with pytest.raises(ValueError, match="lr"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_outcome_weight() -> None:
    cfg = PrmTrainingConfig(
        dataset_path="/tmp/x.jsonl", multi_task_outcome_weight=1.5
    )
    with pytest.raises(ValueError, match="multi_task_outcome_weight"):
        build_training_args(cfg)


def test_build_training_args_rejects_invalid_max_steps() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl", max_steps_per_trace=0)
    with pytest.raises(ValueError, match="max_steps_per_trace"):
        build_training_args(cfg)


# ── build_prm_trainer raises GpuPathNotImplemented ─────────────────


def test_build_prm_trainer_raises_in_30c() -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl")
    with pytest.raises(GpuPathNotImplemented, match="30.F"):
        build_prm_trainer(cfg)


# ── write_run_card ─────────────────────────────────────────────────


def test_write_run_card_persists_payload(tmp_path: Path) -> None:
    cfg = PrmTrainingConfig(dataset_path="/tmp/x.jsonl")
    status = DependencyStatus(available=("torch",), missing=("trl",))
    target = write_run_card(tmp_path, cfg, status)
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["config"]["base_model"] == DEFAULT_BASE_MODEL
    assert payload["dependencies"]["is_satisfied"] is False
    assert payload["schema_version"] == "v0.1.0"


# ── per_step_target_tensor (torch shape) ───────────────────────────


def test_per_step_target_tensor_shape() -> None:
    pytest.importorskip("torch")
    rows = [_trace((0.5, 0.7)), _trace((0.8, 0.9, 0.6))]
    out = per_step_target_tensor(rows, max_steps=4)
    assert out["targets"].shape == (2, 4)
    assert out["mask"].shape == (2, 4)
    # Row 0 has 2 valid steps; row 1 has 3.
    assert out["mask"][0].tolist() == [True, True, False, False]
    assert out["mask"][1].tolist() == [True, True, True, False]


def test_per_step_target_tensor_values() -> None:
    pytest.importorskip("torch")
    rows = [_trace((0.4, 0.6))]
    out = per_step_target_tensor(rows, max_steps=3)
    targets = out["targets"][0].tolist()
    assert targets[0] == pytest.approx(0.4)
    assert targets[1] == pytest.approx(0.6)
    assert targets[2] == pytest.approx(0.0)  # padded


def test_per_step_target_tensor_empty_batch() -> None:
    pytest.importorskip("torch")
    out = per_step_target_tensor([], max_steps=3)
    assert out["targets"].shape == (0, 3)
    assert out["mask"].shape == (0, 3)


def test_per_step_target_tensor_truncates_long_rows() -> None:
    pytest.importorskip("torch")
    rows = [_trace((0.1, 0.2, 0.3, 0.4, 0.5))]
    out = per_step_target_tensor(rows, max_steps=3)
    assert out["targets"].shape == (1, 3)
    assert out["mask"][0].tolist() == [True, True, True]


def test_per_step_target_tensor_rejects_invalid_max_steps() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="max_steps"):
        per_step_target_tensor([_trace((0.5,))], max_steps=0)


# ── per_step_outcome_tensor (D4-D head target) ─────────────────────


def test_per_step_outcome_tensor_shape() -> None:
    pytest.importorskip("torch")
    rows = [_trace((0.4, 0.6)), _trace((0.5, 0.5, 0.7))]
    out = per_step_outcome_tensor(rows)
    assert out.shape == (2,)


def test_per_step_outcome_tensor_empty() -> None:
    pytest.importorskip("torch")
    out = per_step_outcome_tensor([])
    assert out.shape == (0,)
