"""TRL-based trainer wrapper with per-step regression head (Phase 30.C scaffold).

Per :doc:`PHASE_30_PLAN.md` §5 D3-A + D4-A:

- **Student:** Qwen2.5-1.5B-Instruct + per-step regression head.
- **LoRA:** rank 16 / alpha 32 on the attention `q_proj`, `k_proj`,
  `v_proj`, `o_proj` matrices (reuses Phase 29
  :class:`vlabs_reward_train.lora_config.LoraSpec` verbatim).
- **Loss:** per-step MSE on consensus_step_reward (D4-A primary).
  Optional D4-D multi-task add-on with the Phase 29 outcome head
  (see :mod:`verifiable_labs_envs.process_reward.multi_task`).

30.C ships the **scaffolding only** — :func:`train_step` raises
:class:`GpuPathNotImplemented` because the trained student arrives in
30.F when GPU credits resolve. The :class:`PrmTrainingConfig` and
:func:`build_training_args` are fully-typed and round-trippable so they
can be exercised in CI without a GPU.

The real path lights up in 30.F:

```python
trainer = build_prm_trainer(config)
trainer.train()
```

Until then, callers in tests / CI use :func:`validate_dependencies`
to confirm the toolchain is present.

TRL 1.4 API notes (Phase 29.F / 30.F prep, May 2026)
----------------------------------------------------
TRL 1.4 ``GRPOConfig`` renamed two kwargs that the 30.C scaffold had
been carrying under the older TRL surface:

* ``max_prompt_length`` was **removed**; the prompt budget is now
  rolled into ``vllm_max_model_length`` (single budget for prompt +
  trace-completion in the colocate vLLM engine).
* ``kl_coefficient`` was **renamed** to ``beta``. TRL 1.4 defaults
  ``beta`` to ``0.0`` (KL term off); we keep the plan-stated ``0.04``
  to retain KL regularisation on policy drift.

Mirrors the 29.C rename applied to
:mod:`vlabs_reward_train.trainer`; kept in lockstep so the two
configs share the same TRL 1.4 surface.
"""
from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiable_labs_envs.process_reward.dataset import (
    DEFAULT_MAX_STEPS,
    ProcessRewardTraceRow,
)

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
"""D3-A: locked student base model (same as Phase 29)."""

DEFAULT_LR: float = 1e-4
"""Lower LR than Phase 29 (2e-4) — denser per-step supervision needs
gentler steps to avoid oscillation under the multi-task loss."""

DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 8
"""Smaller batch than Phase 29 (16) — per-step traces are denser, so
the effective per-step batch is still ~64-80."""

DEFAULT_GRAD_ACCUM: int = 8
DEFAULT_VLLM_MAX_MODEL_LENGTH: int = 6144
"""TRL 1.4 + vLLM 0.21 single-budget replacement for the legacy
``max_prompt_length``: total prompt + trace length the colocate vLLM
engine must support. Plan calls for 2048 prompt + 4096 trace → 6144.
"""
DEFAULT_MAX_TRACE_LENGTH: int = 4096
"""Per :doc:`PHASE_30_PLAN.md` §8 — traces are longer than completions."""

DEFAULT_NUM_GENERATIONS: int = 4
DEFAULT_BETA: float = 0.04
"""TRL 1.4 rename of ``kl_coefficient`` → ``beta``. TRL 1.4 itself
defaults to ``0.0`` (no KL penalty); we retain the plan-stated
``0.04`` to keep policy-drift regularisation on by default.
"""
DEFAULT_BF16: bool = True
DEFAULT_LORA_R: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05

REQUIRED_DEPS: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
)


@dataclass(frozen=True)
class PrmTrainingConfig:
    """Per-experiment training hyperparameters for the PRM run.

    Defaults are the locked starting points from
    :doc:`PHASE_30_PLAN.md` §8; ablations in 30.F override via the CLI.
    Round-trips through :meth:`to_dict` / :meth:`from_dict` so the W&B
    run-config and the on-disk ``training_config.json`` are bit-stable.
    """

    base_model: str = DEFAULT_BASE_MODEL
    base_rm_checkpoint: str | None = None
    """D13-C shared backbone path: when set, the PRM trainer loads
    LoRA adapters from the Phase 29 distilled RM checkpoint as the
    starting point. ``None`` is the D13-A independent serving path
    (v0.0.1 default)."""

    output_dir: str = "runs/prm-train/exp_001"
    dataset_path: str = ""
    eval_dataset_path: str | None = None
    calib_dataset_path: str | None = None
    lr: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum: int = DEFAULT_GRAD_ACCUM
    vllm_max_model_length: int = DEFAULT_VLLM_MAX_MODEL_LENGTH
    max_trace_length: int = DEFAULT_MAX_TRACE_LENGTH
    max_steps_per_trace: int = DEFAULT_MAX_STEPS
    num_generations: int = DEFAULT_NUM_GENERATIONS
    beta: float = DEFAULT_BETA
    bf16: bool = DEFAULT_BF16
    seed: int = 0
    wandb_project: str = "vlabs-prm-distillation"
    wandb_mode: str = "offline"
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    multi_task: bool = False
    """D4-D toggle: when ``True``, the trainer adds a Phase 29-style
    outcome-level head and trains both heads jointly with the
    multi-task loss configured in
    :mod:`verifiable_labs_envs.process_reward.multi_task`."""

    multi_task_outcome_weight: float = 0.3
    """Weight on the outcome head in the joint loss
    (per-step weight = 1 - this). Defaults to 0.3 per §5 D13-C."""

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrmTrainingConfig:
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in payload:
                kwargs[f.name] = payload[f.name]
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> PrmTrainingConfig:
        kwargs = dataclasses.asdict(self)
        kwargs.update(overrides)
        return PrmTrainingConfig(**kwargs)

    @property
    def shared_backbone(self) -> bool:
        """Predicate: is the trainer running under the D13-B/C shared-
        backbone path?"""
        return self.base_rm_checkpoint is not None

    @property
    def per_step_loss_weight(self) -> float:
        """The per-step loss weight in the multi-task blend
        (always ``1 - multi_task_outcome_weight``)."""
        return 1.0 - float(self.multi_task_outcome_weight)


@dataclass(frozen=True)
class DependencyStatus:
    """Result of :func:`validate_dependencies`."""

    available: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_satisfied(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": list(self.available),
            "missing": list(self.missing),
            "is_satisfied": self.is_satisfied,
        }


def validate_dependencies(
    required: tuple[str, ...] = REQUIRED_DEPS,
) -> DependencyStatus:
    """Probe the runtime for the GPU-training toolchain.

    Same shape as Phase 29 :func:`vlabs_reward_train.trainer.validate_dependencies`
    — the CLI's ``train`` command refuses to proceed on a non-empty
    ``missing`` tuple; ``dry-run`` reports the status but proceeds.
    """
    available: list[str] = []
    missing: list[str] = []
    for dep in required:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
        else:
            available.append(dep)
    return DependencyStatus(available=tuple(available), missing=tuple(missing))


def build_training_args(config: PrmTrainingConfig) -> dict[str, Any]:
    """Build the keyword-arg dict for the TRL 1.4 ``GRPOConfig`` constructor
    (or whichever TRL trainer 30.F targets).

    Returned as a plain dict so it's serialisable + diffable in tests
    even when TRL isn't installed. The 30.F training step calls the
    real TRL config with ``GRPOConfig(**build_training_args(config))``.

    Key shape (TRL 1.4):

    * ``beta`` — KL coefficient (renamed from ``kl_coefficient``).
    * ``vllm_max_model_length`` — single prompt+trace length budget
      for the colocate vLLM engine (replaces ``max_prompt_length``).
    * ``max_completion_length`` — generation-side cap on the trace,
      sourced from ``config.max_trace_length``.
    """
    if not config.dataset_path:
        raise ValueError("dataset_path must be set on the PrmTrainingConfig")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive; got {config.epochs}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {config.batch_size}")
    if not 0.0 < config.lr < 1.0:
        raise ValueError(f"lr must be in (0, 1); got {config.lr}")
    if not 0.0 <= config.multi_task_outcome_weight <= 1.0:
        raise ValueError(
            "multi_task_outcome_weight must be in [0, 1]; "
            f"got {config.multi_task_outcome_weight}"
        )
    if config.max_steps_per_trace <= 0:
        raise ValueError(
            f"max_steps_per_trace must be positive; got {config.max_steps_per_trace}"
        )

    return {
        "output_dir": config.output_dir,
        "learning_rate": config.lr,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "vllm_max_model_length": config.vllm_max_model_length,
        "max_completion_length": config.max_trace_length,
        "num_generations": config.num_generations,
        "beta": config.beta,
        "bf16": config.bf16,
        "seed": config.seed,
        "report_to": (
            ["wandb"] if config.wandb_mode in ("online", "offline") else []
        ),
        "logging_steps": 10,
        "save_steps": 50,
    }


class GpuPathNotImplemented(RuntimeError):
    """Raised when 30.C-only code attempts to invoke the 30.F GPU path."""


def build_prm_trainer(config: PrmTrainingConfig) -> Any:
    """Construct a TRL trainer wired to the PRM with per-step head.
    **Not implemented in 30.C** — the production path lights up in 30.F
    when GPU credits resolve. Tests assert this raises
    :class:`GpuPathNotImplemented`."""
    del config
    raise GpuPathNotImplemented(
        "GPU training arrives in Phase 30.F. Until then, use "
        "`vlabs-prm-train dry-run` to inspect the resolved config."
    )


def write_run_card(
    output_dir: Path | str,
    config: PrmTrainingConfig,
    status: DependencyStatus,
) -> Path:
    """Persist a ``run_card.json`` describing the resolved config +
    dep status. Used by 30.D's eval harness to discover which configs
    produced which checkpoints."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    target = p / "run_card.json"
    payload = {
        "config": config.to_dict(),
        "dependencies": status.to_dict(),
        "schema_version": "v0.1.0",
    }
    import json  # noqa: PLC0415

    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


# ── per-step head shape utility (CPU-only Tensor manipulation) ──────


def per_step_target_tensor(
    rows: Sequence[ProcessRewardTraceRow],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    pad_value: float = 0.0,
) -> Any:
    """Pack the per-step consensus rewards across a batch of rows into
    a ``(batch_size, max_steps)`` tensor + matching mask.

    Rows shorter than ``max_steps`` are right-padded with ``pad_value``;
    the boolean mask marks valid step positions for the loss to ignore
    padding. Lazy torch import — callers in CPU-only paths don't pay
    the cost.

    Returns a dict ``{"targets": tensor, "mask": tensor}`` so the
    trainer can apply masked-MSE in one line:
    ``loss = ((preds - targets) ** 2 * mask).sum() / mask.sum()``.
    """
    import torch  # noqa: PLC0415

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive; got {max_steps}")
    if not rows:
        return {
            "targets": torch.zeros((0, max_steps), dtype=torch.float32),
            "mask": torch.zeros((0, max_steps), dtype=torch.bool),
        }

    batch = len(rows)
    targets = torch.full(
        (batch, max_steps), pad_value, dtype=torch.float32
    )
    mask = torch.zeros((batch, max_steps), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = min(row.step_count, max_steps)
        for t in range(n):
            targets[i, t] = float(row.step_consensus_rewards[t])
            mask[i, t] = True
    return {"targets": targets, "mask": mask}


def per_step_outcome_tensor(rows: Sequence[ProcessRewardTraceRow]) -> Any:
    """Aggregate-target tensor for the D4-D outcome head — shape
    ``(batch_size,)``."""
    import torch  # noqa: PLC0415

    if not rows:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(
        [float(r.aggregate_reward) for r in rows], dtype=torch.float32
    )


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_BF16",
    "DEFAULT_EPOCHS",
    "DEFAULT_GRAD_ACCUM",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_DROPOUT",
    "DEFAULT_LORA_R",
    "DEFAULT_LR",
    "DEFAULT_MAX_TRACE_LENGTH",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_VLLM_MAX_MODEL_LENGTH",
    "REQUIRED_DEPS",
    "DependencyStatus",
    "GpuPathNotImplemented",
    "PrmTrainingConfig",
    "build_prm_trainer",
    "build_training_args",
    "per_step_outcome_tensor",
    "per_step_target_tensor",
    "validate_dependencies",
    "write_run_card",
]
