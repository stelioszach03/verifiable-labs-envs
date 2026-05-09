"""JSONL → torch Dataset / DataLoader for reward distillation training.

Phase 29.C ships the **CPU-only** harness — the dataset emits
``(prompt: str, completion: str, target: float)`` tuples plus the
optional ``conformal_interval``. Tokenisation lives in 29.F's actual
training step where the chosen tokenizer (Qwen2.5-1.5B-Instruct) and
device move can pin to a real GPU.

Key design points:

- The :class:`RewardDataset` is a thin wrapper over a list of
  :class:`RewardTrainingRow` objects, so it's exchangeable for any
  iterable of rows; tests pass synthetic rows via
  :func:`build_synthetic_rows`.
- :func:`build_dataloader` lazily imports ``torch.utils.data.DataLoader``
  but the ``RewardDataset`` itself is pure-Python so help text /
  dry-run paths don't need torch.
- :func:`collate_reward_batch` returns a dict of equal-length lists +
  a ``targets`` ``torch.Tensor`` so the trainer can move just that
  tensor to the GPU.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifiable_labs_envs.reward_distillation.dataset import (
    RewardTrainingRow,
    read_jsonl,
)

DEFAULT_BATCH_SIZE: int = 16
DEFAULT_NUM_WORKERS: int = 0
"""DataLoader workers default to 0 because tests spawn under pytest's
collection phase; multi-process workers + Python's import lock
deadlock easily there. The 29.F training script overrides via the
CLI flag."""


@dataclass(frozen=True)
class RewardTrainingExample:
    """Single training example shape — what the trainer sees per row.

    ``target`` is the consensus reward (the MSE loss target, D6-A).
    ``conformal_low`` / ``conformal_high`` are the env's CI bounds on
    the env reward; ``None`` for external rows that have no env signal.
    """

    prompt: str
    completion: str
    target: float
    conformal_low: float | None
    conformal_high: float | None
    env_id: str | None
    source: str
    row_id: str

    @classmethod
    def from_row(cls, row: RewardTrainingRow) -> RewardTrainingExample:
        ci = row.conformal_interval
        return cls(
            prompt=row.prompt,
            completion=row.completion,
            target=float(row.consensus_reward),
            conformal_low=float(ci[0]) if ci is not None else None,
            conformal_high=float(ci[1]) if ci is not None else None,
            env_id=row.env_id,
            source=row.source,
            row_id=row.row_id,
        )


class RewardDataset:
    """Map-style dataset wrapping a list of :class:`RewardTrainingRow`.

    Pure-Python; **does not** import torch. Use
    :func:`build_dataloader` to wrap into a torch DataLoader for the
    training step.
    """

    def __init__(self, rows: Sequence[RewardTrainingRow]) -> None:
        self._rows: list[RewardTrainingRow] = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> RewardTrainingExample:
        return RewardTrainingExample.from_row(self._rows[index])

    def __iter__(self) -> Iterable[RewardTrainingExample]:
        for row in self._rows:
            yield RewardTrainingExample.from_row(row)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> RewardDataset:
        return cls(read_jsonl(path))

    def split(
        self,
        validation_fraction: float = 0.1,
        *,
        seed: int = 0,
    ) -> tuple[RewardDataset, RewardDataset]:
        """Deterministic train/val split. Reuses the row order so that
        with the same seed the split is bit-stable across runs.

        ``validation_fraction == 0`` returns ``(self, empty_dataset)``.
        """
        if not 0.0 <= validation_fraction <= 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1]; got {validation_fraction}"
            )
        n = len(self._rows)
        if validation_fraction == 0.0 or n == 0:
            return RewardDataset(self._rows), RewardDataset([])

        import numpy as np  # noqa: PLC0415 — lazy

        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_val = max(1, int(round(n * validation_fraction))) if n > 1 else 0
        val_idx = sorted(int(i) for i in indices[:n_val])
        train_idx = sorted(int(i) for i in indices[n_val:])
        return (
            RewardDataset([self._rows[i] for i in train_idx]),
            RewardDataset([self._rows[i] for i in val_idx]),
        )


def build_synthetic_rows(n: int = 8, *, seed: int = 0) -> list[RewardTrainingRow]:
    """Deterministic stand-in rows for unit tests and dry-runs.

    Each row carries a fixed ``env_id="math-algebra"`` and
    ``source="env"``; the ``consensus_reward`` is sampled uniformly so
    the synthetic distribution exercises the calibration math too.
    """
    import numpy as np  # noqa: PLC0415

    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    rng = np.random.default_rng(seed)
    rows: list[RewardTrainingRow] = []
    for i in range(n):
        reward = float(rng.uniform(0.0, 1.0))
        rows.append(
            RewardTrainingRow(
                row_id=f"rwd_synth_{i:08x}",
                env_id="math-algebra",
                prompt=f"synthetic prompt {i}",
                completion=f"synthetic completion {i}",
                env_reward=reward,
                env_components={"correct": reward},
                conformal_interval=(max(0.0, reward - 0.1), min(1.0, reward + 0.1)),
                frontier_judgment=None,
                frontier_rationale=None,
                consensus_reward=reward,
                disagreement=None,
                source="env",
                metadata={"synthetic": True},
            )
        )
    return rows


def collate_reward_batch(batch: Sequence[RewardTrainingExample]) -> dict[str, Any]:
    """Stack a list of examples into a per-key dict.

    Texts (``prompts``, ``completions``) become parallel lists for the
    tokenizer in the training step; ``targets`` is a 1-D ``torch.Tensor``
    so the trainer can move just that tensor to the GPU. The conformal
    bounds are optional — when present they're stacked as 1-D Tensors,
    when absent they're None.
    """
    import torch  # noqa: PLC0415 — lazy

    prompts = [ex.prompt for ex in batch]
    completions = [ex.completion for ex in batch]
    targets = torch.tensor([ex.target for ex in batch], dtype=torch.float32)

    has_lows = all(ex.conformal_low is not None for ex in batch)
    has_highs = all(ex.conformal_high is not None for ex in batch)
    conformal_low = (
        torch.tensor([ex.conformal_low for ex in batch], dtype=torch.float32)
        if has_lows
        else None
    )
    conformal_high = (
        torch.tensor([ex.conformal_high for ex in batch], dtype=torch.float32)
        if has_highs
        else None
    )
    return {
        "prompts": prompts,
        "completions": completions,
        "targets": targets,
        "conformal_low": conformal_low,
        "conformal_high": conformal_high,
        "row_ids": [ex.row_id for ex in batch],
        "env_ids": [ex.env_id for ex in batch],
        "sources": [ex.source for ex in batch],
    }


def build_dataloader(
    dataset: RewardDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
    collate_fn: Callable[[Sequence[RewardTrainingExample]], dict[str, Any]] | None = None,
) -> Any:
    """Wrap a :class:`RewardDataset` in a ``torch.utils.data.DataLoader``.

    Lazy torch import — callers in CPU-only paths don't pay the cost.
    The default ``collate_fn`` is :func:`collate_reward_batch`.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative; got {num_workers}")
    if len(dataset) == 0:
        raise ValueError("dataset is empty; cannot build a DataLoader")

    from torch.utils.data import DataLoader  # noqa: PLC0415

    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or collate_reward_batch,
    )


def dataset_target_stats(dataset: RewardDataset) -> dict[str, float]:
    """Aggregate target stats over the dataset; emitted at training
    start so the run log records the dataset shape."""
    if len(dataset) == 0:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    import numpy as np  # noqa: PLC0415

    targets = np.asarray([ex.target for ex in dataset], dtype=np.float64)
    return {
        "count": float(targets.size),
        "mean": float(targets.mean()),
        "min": float(targets.min()),
        "max": float(targets.max()),
        "std": float(targets.std(ddof=0)),
    }


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "RewardDataset",
    "RewardTrainingExample",
    "build_dataloader",
    "build_synthetic_rows",
    "collate_reward_batch",
    "dataset_target_stats",
]
