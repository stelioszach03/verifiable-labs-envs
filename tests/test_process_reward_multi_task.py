"""Tests for ``verifiable_labs_envs.process_reward.multi_task``."""
from __future__ import annotations

import pytest

from verifiable_labs_envs.process_reward.dataset import (
    SCHEMA_VERSION,
    ProcessRewardTraceRow,
)
from verifiable_labs_envs.process_reward.multi_task import (
    DEFAULT_OUTCOME_WEIGHT,
    DEFAULT_PER_STEP_WEIGHT,
    MultiTaskConfig,
    build_multi_task_loss,
    is_shared_backbone_ready,
    loss_summary,
    split_phase29_compat_payload,
)


def _trace(env_rewards: tuple[float, ...]) -> ProcessRewardTraceRow:
    n = len(env_rewards)
    return ProcessRewardTraceRow(
        row_id="prw_x",
        env_id="math-algebra",
        prompt="p",
        steps=tuple(f"s{i}" for i in range(n)),
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


# ── locked weights ────────────────────────────────────────────────


def test_default_weights_locked_per_plan() -> None:
    """Plan §5 D4-D / D13-C: per-step 0.7 + outcome 0.3."""
    assert pytest.approx(0.3) == DEFAULT_OUTCOME_WEIGHT
    assert pytest.approx(0.7) == DEFAULT_PER_STEP_WEIGHT
    assert pytest.approx(1.0) == DEFAULT_PER_STEP_WEIGHT + DEFAULT_OUTCOME_WEIGHT


# ── MultiTaskConfig ────────────────────────────────────────────────


def test_config_round_trip() -> None:
    cfg = MultiTaskConfig(
        per_step_weight=0.6, outcome_weight=0.4, share_backbone=True, enable=True
    )
    restored = MultiTaskConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_config_normalised_sums_to_one() -> None:
    cfg = MultiTaskConfig(per_step_weight=2.0, outcome_weight=2.0)
    norm = cfg.normalised()
    assert norm.per_step_weight + norm.outcome_weight == pytest.approx(1.0)


def test_config_normalised_rejects_zero_total() -> None:
    cfg = MultiTaskConfig(per_step_weight=0.0, outcome_weight=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        cfg.normalised()


def test_config_default_disabled() -> None:
    """Default is enable=False so plain per-step training is the
    out-of-the-box path."""
    cfg = MultiTaskConfig()
    assert cfg.enable is False
    assert cfg.share_backbone is False


def test_config_freeze_outcome_head_default_false() -> None:
    cfg = MultiTaskConfig()
    assert cfg.freeze_outcome_head is False


# ── build_multi_task_loss ──────────────────────────────────────────


def test_loss_per_step_only_when_disabled() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = MultiTaskConfig(enable=False, per_step_weight=1.0)
    loss_fn = build_multi_task_loss(cfg)
    preds = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    targets = torch.tensor([[0.5, 1.0], [0.0, 0.5]])
    mask = torch.tensor([[True, True], [True, True]])
    loss = loss_fn(preds, targets, mask)
    # MSE values: (0)^2 + (0.5)^2 + (0.5)^2 + (0)^2 = 0.5; mean = 0.125.
    assert float(loss) == pytest.approx(0.125)


def test_loss_blends_when_enabled() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = MultiTaskConfig(per_step_weight=0.7, outcome_weight=0.3, enable=True)
    loss_fn = build_multi_task_loss(cfg)
    preds = torch.tensor([[0.5, 0.5]])
    targets = torch.tensor([[0.5, 0.5]])  # zero per-step loss
    mask = torch.tensor([[True, True]])
    outcome_preds = torch.tensor([0.0])
    outcome_targets = torch.tensor([1.0])  # outcome MSE = 1
    loss = loss_fn(preds, targets, mask, outcome_preds, outcome_targets)
    # 0.7*0 + 0.3*1 = 0.3.
    assert float(loss) == pytest.approx(0.3)


def test_loss_masks_invalid_positions() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = MultiTaskConfig(enable=False, per_step_weight=1.0)
    loss_fn = build_multi_task_loss(cfg)
    preds = torch.tensor([[0.5, 99.0]])  # second position is "padding"
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = loss_fn(preds, targets, mask)
    # Only step 0 contributes: (0.5 - 1.0)^2 = 0.25 / 1 valid pos = 0.25.
    assert float(loss) == pytest.approx(0.25)


def test_loss_requires_outcome_when_enabled() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = MultiTaskConfig(enable=True)
    loss_fn = build_multi_task_loss(cfg)
    preds = torch.tensor([[0.5]])
    targets = torch.tensor([[0.5]])
    mask = torch.tensor([[True]])
    with pytest.raises(ValueError, match="outcome_preds"):
        loss_fn(preds, targets, mask)


def test_loss_zero_when_no_valid_positions() -> None:
    """All-mask-False batch — divisor clamp keeps loss at 0 (not NaN)."""
    pytest.importorskip("torch")
    import torch

    cfg = MultiTaskConfig(enable=False)
    loss_fn = build_multi_task_loss(cfg)
    preds = torch.tensor([[0.5, 0.5]])
    targets = torch.tensor([[0.5, 0.5]])
    mask = torch.tensor([[False, False]])
    loss = loss_fn(preds, targets, mask)
    assert float(loss) == pytest.approx(0.0)


# ── split_phase29_compat_payload ───────────────────────────────────


def test_split_payload_shapes() -> None:
    pytest.importorskip("torch")
    rows = [_trace((0.5, 0.7)), _trace((0.6, 0.8, 0.9))]
    payload = split_phase29_compat_payload(rows)
    assert payload["per_step_targets"].shape == (2, 32)
    assert payload["per_step_mask"].shape == (2, 32)
    assert payload["outcome_targets"].shape == (2,)
    assert len(payload["prompts"]) == 2
    assert len(payload["traces_joined"]) == 2


def test_split_payload_traces_joined_uses_newlines() -> None:
    rows = [_trace((0.5, 0.7))]
    payload = split_phase29_compat_payload(rows)
    assert "\n" in payload["traces_joined"][0]


def test_split_payload_empty_rows() -> None:
    pytest.importorskip("torch")
    payload = split_phase29_compat_payload([])
    assert payload["per_step_targets"].shape == (0, 32)
    assert payload["prompts"] == []


# ── helpers ────────────────────────────────────────────────────────


def test_is_shared_backbone_ready_predicate() -> None:
    assert is_shared_backbone_ready(None) is False
    assert is_shared_backbone_ready("") is False
    assert is_shared_backbone_ready("runs/reward-train/exp_004/") is True


def test_loss_summary_disabled() -> None:
    cfg = MultiTaskConfig(enable=False)
    summary = loss_summary(cfg)
    assert summary["enable"] is False
    # Disabled → outcome share normalises to 0.
    assert summary["outcome_weight_normalised"] == pytest.approx(0.0)
    assert summary["per_step_weight_normalised"] == pytest.approx(1.0)


def test_loss_summary_enabled_normalised() -> None:
    cfg = MultiTaskConfig(per_step_weight=0.7, outcome_weight=0.3, enable=True)
    summary = loss_summary(cfg)
    assert summary["per_step_weight_normalised"] == pytest.approx(0.7)
    assert summary["outcome_weight_normalised"] == pytest.approx(0.3)
