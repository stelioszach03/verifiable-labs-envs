"""TRL ``GRPOTrainer`` wrapper for the distilled reward model (29.C scaffold).

29.C ships the **scaffolding only** — `train_step` raises
:class:`RuntimeError` because the trained student arrives in 29.F when
GPU credits resolve. The :class:`TrainingConfig` and
:func:`build_training_args` are fully-typed and round-trippable so
they can be exercised in CI without a GPU.

The real path lights up in 29.F:

```python
trainer = build_grpo_trainer(config)
trainer.train()
```

Until then, callers in tests / CI use :func:`validate_dependencies`
to confirm the toolchain is present.

TRL 1.4 API notes (Phase 29.F prep, May 2026)
---------------------------------------------
TRL 1.4 ``GRPOConfig`` renamed two kwargs that the 29.C scaffold had
been carrying under the older TRL surface:

* ``max_prompt_length`` was **removed**; the prompt budget is now
  rolled into ``vllm_max_model_length`` (single budget for prompt +
  completion in the colocate vLLM engine).
* ``kl_coefficient`` was **renamed** to ``beta``. TRL 1.4 defaults
  ``beta`` to ``0.0`` (KL term off); we keep the plan-stated ``0.04``
  to retain KL regularisation on policy drift.

The on-disk ``run_card.json`` / ``training_config.json`` therefore
use the new field names — this is a hard schema break vs. earlier
29.C run cards. There were no production run cards in 29.C, so
nothing on disk needs migrating.
"""
from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlabs_reward_train.lora_config import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    LoraSpec,
)

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
"""D2-A: the locked student base model."""

DEFAULT_LR: float = 2e-4
DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_GRAD_ACCUM: int = 4
DEFAULT_VLLM_MAX_MODEL_LENGTH: int = 3072
"""TRL 1.4 + vLLM 0.21 single-budget replacement for the legacy
``max_prompt_length``: total prompt + completion length the colocate
vLLM engine must support. The plan calls for 2048 prompt + 1024
completion → 3072.
"""
DEFAULT_MAX_COMPLETION_LENGTH: int = 1024
DEFAULT_NUM_GENERATIONS: int = 4
DEFAULT_BETA: float = 0.04
"""TRL 1.4 rename of ``kl_coefficient`` → ``beta``. TRL 1.4 itself
defaults to ``0.0`` (no KL penalty); we retain the plan-stated
``0.04`` to keep policy-drift regularisation on by default.
"""
DEFAULT_BF16: bool = True

REQUIRED_DEPS: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Per-experiment training hyperparameters.

    Defaults are the locked starting points from CLAUDE.md; ablations
    in 29.F override via the CLI flags. Round-trips through
    :meth:`to_dict` / :meth:`from_dict` so the W&B run-config and the
    on-disk ``training_config.json`` are bit-stable.
    """

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = "runs/reward-train/exp_001"
    dataset_path: str = ""
    eval_dataset_path: str | None = None
    calib_dataset_path: str | None = None
    lr: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum: int = DEFAULT_GRAD_ACCUM
    vllm_max_model_length: int = DEFAULT_VLLM_MAX_MODEL_LENGTH
    max_completion_length: int = DEFAULT_MAX_COMPLETION_LENGTH
    num_generations: int = DEFAULT_NUM_GENERATIONS
    beta: float = DEFAULT_BETA
    bf16: bool = DEFAULT_BF16
    seed: int = 0
    wandb_project: str = "vlabs-reward-distillation"
    wandb_mode: str = "offline"
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def lora_spec(self) -> LoraSpec:
        return LoraSpec(r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingConfig:
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in payload:
                kwargs[f.name] = payload[f.name]
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> TrainingConfig:
        kwargs = dataclasses.asdict(self)
        kwargs.update(overrides)
        return TrainingConfig(**kwargs)


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

    Imports each package lazily and records the missing ones. The CLI's
    ``train`` command refuses to proceed on a non-empty ``missing``
    tuple; ``dry-run`` reports the status but proceeds.
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


def build_training_args(config: TrainingConfig) -> dict[str, Any]:
    """Build the keyword-arg dict for the TRL 1.4 ``GRPOConfig`` constructor.

    Returned as a plain dict so it's serialisable + diff-able in tests
    even when TRL isn't installed. The 29.F training step calls
    ``GRPOConfig(**build_training_args(config))``.

    Key shape (TRL 1.4):

    * ``beta`` — KL coefficient (renamed from ``kl_coefficient``).
    * ``vllm_max_model_length`` — single prompt+completion length
      budget for the colocate vLLM engine (replaces
      ``max_prompt_length``).
    * ``max_completion_length`` — generation-side cap, unchanged.
    """
    if not config.dataset_path:
        raise ValueError("dataset_path must be set on the TrainingConfig")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive; got {config.epochs}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {config.batch_size}")
    if not 0.0 < config.lr < 1.0:
        raise ValueError(f"lr must be in (0, 1); got {config.lr}")

    return {
        "output_dir": config.output_dir,
        "learning_rate": config.lr,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "vllm_max_model_length": config.vllm_max_model_length,
        "max_completion_length": config.max_completion_length,
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
    """Raised when 29.C-only code attempts to invoke the 29.F GPU path."""


def build_grpo_trainer(config: TrainingConfig) -> Any:
    """Construct a TRL ``GRPOTrainer`` wired to the distilled reward
    model. **Not implemented in 29.C** — the production path lights up
    in 29.F when GPU credits resolve. Tests assert this raises
    :class:`GpuPathNotImplemented`.
    """
    del config
    raise GpuPathNotImplemented(
        "GPU training arrives in Phase 29.F. Until then, use "
        "`vlabs-reward-train dry-run` to inspect the resolved config."
    )


def write_run_card(
    output_dir: Path | str, config: TrainingConfig, status: DependencyStatus
) -> Path:
    """Persist a ``run_card.json`` describing the resolved config +
    dep status. Used by 29.D's eval harness to discover which configs
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


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_BF16",
    "DEFAULT_EPOCHS",
    "DEFAULT_GRAD_ACCUM",
    "DEFAULT_LR",
    "DEFAULT_MAX_COMPLETION_LENGTH",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_VLLM_MAX_MODEL_LENGTH",
    "REQUIRED_DEPS",
    "DependencyStatus",
    "GpuPathNotImplemented",
    "TrainingConfig",
    "build_grpo_trainer",
    "build_training_args",
    "validate_dependencies",
    "write_run_card",
]
