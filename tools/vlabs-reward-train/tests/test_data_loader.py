"""Tests for ``vlabs_reward_train.data_loader``."""
from __future__ import annotations

from pathlib import Path

import pytest
from verifiable_labs_envs.reward_distillation.dataset import write_jsonl

from vlabs_reward_train.data_loader import (
    DEFAULT_BATCH_SIZE,
    RewardDataset,
    RewardTrainingExample,
    build_dataloader,
    build_synthetic_rows,
    collate_reward_batch,
    dataset_target_stats,
)


def test_build_synthetic_rows_basic() -> None:
    rows = build_synthetic_rows(5)
    assert len(rows) == 5
    for r in rows:
        assert r.metadata["synthetic"] is True
        assert 0.0 <= r.consensus_reward <= 1.0


def test_build_synthetic_rows_deterministic() -> None:
    a = build_synthetic_rows(3, seed=42)
    b = build_synthetic_rows(3, seed=42)
    assert [r.row_id for r in a] == [r.row_id for r in b]
    assert [r.consensus_reward for r in a] == [r.consensus_reward for r in b]


def test_build_synthetic_rows_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_synthetic_rows(-1)


def test_reward_dataset_basic() -> None:
    rows = build_synthetic_rows(4)
    ds = RewardDataset(rows)
    assert len(ds) == 4
    ex = ds[0]
    assert isinstance(ex, RewardTrainingExample)
    assert ex.row_id == rows[0].row_id


def test_reward_dataset_iter_yields_examples() -> None:
    rows = build_synthetic_rows(3)
    ds = RewardDataset(rows)
    examples = list(ds)
    assert len(examples) == 3
    assert all(isinstance(e, RewardTrainingExample) for e in examples)


def test_reward_dataset_split_deterministic() -> None:
    rows = build_synthetic_rows(20)
    ds = RewardDataset(rows)
    train_a, val_a = ds.split(0.25, seed=0)
    train_b, val_b = ds.split(0.25, seed=0)
    assert [e.row_id for e in train_a] == [e.row_id for e in train_b]
    assert [e.row_id for e in val_a] == [e.row_id for e in val_b]
    assert len(train_a) + len(val_a) == 20
    assert len(val_a) == 5


def test_reward_dataset_split_zero_validation() -> None:
    ds = RewardDataset(build_synthetic_rows(5))
    train, val = ds.split(0.0)
    assert len(train) == 5
    assert len(val) == 0


def test_reward_dataset_split_rejects_invalid_fraction() -> None:
    ds = RewardDataset(build_synthetic_rows(5))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ds.split(1.5)


def test_reward_dataset_from_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = build_synthetic_rows(6)
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    ds = RewardDataset.from_jsonl(path)
    assert len(ds) == 6
    assert ds[0].row_id == rows[0].row_id


def test_collate_reward_batch_emits_torch_tensor() -> None:
    pytest.importorskip("torch")
    rows = build_synthetic_rows(4)
    ds = RewardDataset(rows)
    batch = [ds[i] for i in range(4)]
    collated = collate_reward_batch(batch)
    assert collated["targets"].shape == (4,)
    assert len(collated["prompts"]) == 4
    assert len(collated["completions"]) == 4
    assert collated["row_ids"] == [r.row_id for r in rows]


def test_build_dataloader_smoke() -> None:
    pytest.importorskip("torch")
    rows = build_synthetic_rows(8)
    ds = RewardDataset(rows)
    loader = build_dataloader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4
    for batch in batches:
        assert batch["targets"].shape == (2,)


def test_build_dataloader_rejects_empty_dataset() -> None:
    ds = RewardDataset([])
    with pytest.raises(ValueError, match="empty"):
        build_dataloader(ds)


def test_build_dataloader_rejects_invalid_args() -> None:
    pytest.importorskip("torch")
    ds = RewardDataset(build_synthetic_rows(2))
    with pytest.raises(ValueError, match="batch_size"):
        build_dataloader(ds, batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        build_dataloader(ds, num_workers=-1)


def test_default_batch_size_locked_per_plan() -> None:
    """Plan §5 D6-A: batch size 16 (effective 64 with grad-accum 4×)."""
    assert DEFAULT_BATCH_SIZE == 16


def test_dataset_target_stats() -> None:
    ds = RewardDataset(build_synthetic_rows(10, seed=0))
    stats = dataset_target_stats(ds)
    assert stats["count"] == 10.0
    assert 0.0 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1.0
    assert stats["std"] >= 0.0


def test_dataset_target_stats_empty() -> None:
    stats = dataset_target_stats(RewardDataset([]))
    assert stats["count"] == 0.0
