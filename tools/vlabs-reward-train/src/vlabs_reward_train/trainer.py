"""TRL ``GRPOTrainer`` wrapper for the distilled reward model (Phase 29.F unlock).

The 29.C scaffold raised :class:`GpuPathNotImplemented` from
:func:`build_grpo_trainer`; 29.F replaces that stub with a live
TRL 1.4 + vLLM 0.21 colocate trainer. The :class:`TrainingConfig`
and :func:`build_training_args` are fully-typed and round-trippable
so CI smoke tests can exercise the config path without a GPU.

The real path now lights up:

```python
trainer = build_grpo_trainer(config)
trainer.train()
```

Callers in tests / CI use :func:`validate_dependencies` to confirm
the toolchain is present before invoking the GPU path.

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

vLLM colocate kwargs (29.F)
---------------------------
Phase 29.F adds first-class vLLM fields to :class:`TrainingConfig`
so the trainer can spin a colocate engine without callers having to
poke at ``extra``:

* ``vllm_gpu_memory_utilization`` — fraction of GPU memory the vLLM
  engine may claim (TRL default 0.3; tunable for 32 GB Blackwell).
* ``vllm_tensor_parallel_size`` — TP degree (default 1 for single-GPU).
* ``vllm_mode`` — "colocate" runs vLLM in-process; "server" starts
  a separate vLLM server (we use "colocate" by default).
* ``env_id`` — the verifiable-labs-envs registry id whose reward
  function drives the policy update.
* ``max_steps`` — hard step cap (used by smoke runs / ablations).

Prompt-building (29.F, Option-A fix)
------------------------------------
GRPO needs the policy to emit answers in the env adapter's *LLM
wire format* (e.g. ``{"support_idx": [...], "support_amp_x1000":
[...]}`` for sparse-fourier-recovery). The reward-distillation
JSONL stores prompts in a more compact extract-pipeline format and
stores completions as the env's internal ``Prediction`` dataclass —
neither is what the LLM-side adapter parses.

:func:`_load_grpo_prompts_dataset` therefore **ignores** the JSONL's
``prompt``/``completion`` fields and regenerates the canonical LLM
prompt from each row's ``metadata.seed`` using:

* ``env.generate_instance(seed=N)`` to get the deterministic env
  instance,
* ``adapter.build_user_prompt(instance)`` + ``adapter.system_prompt``
  to produce the schema-instructing user/system pair,
* ``tokenizer.apply_chat_template(...)`` to wrap them in the Qwen
  chat-template format (so the instruction-tuned base model knows
  to respond as the assistant).

The JSONL is treated as a source of valid ``(env_id, seed)`` pairs
only.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
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

_LOG = logging.getLogger(__name__)

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"
"""D2-A: the locked student base model."""

DEFAULT_LR: float = 2e-4
DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_GRAD_ACCUM: int = 4
DEFAULT_VLLM_MAX_MODEL_LENGTH: int = 3072
"""TRL 1.4 + vLLM 0.21 single-budget replacement for the legacy
``max_prompt_length``: total prompt + completion length the colocate
vLLM engine must support. The adapter's integer-scaled LLM-wire
encoding is compact (≤ 1300 tokens for sparse-fourier-recovery on
n=256, m=200); 2048 prompt + 1024 completion → 3072 leaves ample
headroom.
"""
DEFAULT_MAX_COMPLETION_LENGTH: int = 1024
DEFAULT_NUM_GENERATIONS: int = 4
DEFAULT_BETA: float = 0.04
"""TRL 1.4 rename of ``kl_coefficient`` → ``beta``. TRL 1.4 itself
defaults to ``0.0`` (no KL penalty); we retain the plan-stated
``0.04`` to keep policy-drift regularisation on by default.
"""
DEFAULT_BF16: bool = True
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION: float = 0.3
"""Fraction of GPU memory the colocate vLLM engine may claim. 0.3
leaves headroom on a 32 GB RTX 5090 for the policy gradient pass
+ optimizer state."""
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE: int = 1
"""TP degree for the vLLM engine. Single-GPU default; bump for
multi-GPU pods."""
DEFAULT_VLLM_MODE: str = "colocate"
"""TRL ``vllm_mode``: 'colocate' runs vLLM inside the trainer
process (preferred for single-pod GRPO); 'server' starts an
external vLLM server."""
DEFAULT_ENV_ID: str = "sparse-fourier-recovery"
"""Plan baseline env for the M5/M6 milestone. Override via the
``--env-id`` CLI flag for ablations."""
DEFAULT_MAX_STEPS: int = -1
"""``-1`` means run to the epoch boundary (TRL convention). Set
positive for smoke runs / step-capped ablations."""

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
    max_steps: int = DEFAULT_MAX_STEPS
    """Hard step cap (TRL convention: ``-1`` = run to epoch boundary)."""

    vllm_gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    vllm_tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    vllm_mode: str = DEFAULT_VLLM_MODE
    env_id: str = DEFAULT_ENV_ID
    """Verifiable-labs-envs registry id whose reward function drives
    the policy update (e.g. ``"sparse-fourier-recovery"``)."""

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
    * ``max_steps`` — hard step cap (``-1`` runs to epoch end).
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
        "max_steps": config.max_steps,
        "report_to": (
            ["wandb"] if config.wandb_mode in ("online", "offline") else []
        ),
        "logging_steps": 10,
        "save_steps": 50,
    }


class GpuPathNotImplemented(RuntimeError):
    """Retained for backward compatibility — the 29.F unlock no longer
    raises this from :func:`build_grpo_trainer`, but downstream code
    that catches this exception still compiles."""


def _load_env_and_adapter(env_id: str) -> tuple[Any, Any]:
    """Load env + adapter for ``env_id``, retrying without the
    sparse-Fourier-only ``calibration_quantile`` shortcut on envs
    that don't accept it (matches the pattern used inside
    ``make_reward_fn``)."""
    from verifiable_labs_envs import load_environment  # noqa: PLC0415
    from verifiable_labs_envs.solvers.llm_solver import get_adapter  # noqa: PLC0415

    try:
        env = load_environment(env_id, calibration_quantile=2.0)
    except TypeError:
        env = load_environment(env_id)
    adapter = get_adapter(env_id)
    return env, adapter


def _load_grpo_prompts_dataset(
    jsonl_path: str | Path,
    *,
    env_id: str,
    tokenizer: Any,
    seed_kwarg: str = "instance_seed",
) -> Any:
    """Build a prompts-only HF :class:`datasets.Dataset` suitable for
    :class:`trl.GRPOTrainer`.

    Reads the reward-distillation JSONL but treats it as a source of
    deterministic ``(env_id, seed)`` pairs only. For each matching
    row the canonical LLM prompt is rebuilt from scratch via the env's
    LLM adapter + the tokenizer's chat template:

    1. ``env.generate_instance(seed=md.seed)`` regenerates the env
       instance bit-stably.
    2. ``adapter.build_user_prompt(instance)`` produces the
       schema-instructing user message (e.g. for
       sparse-fourier-recovery, "OUTPUT SCHEMA: {support_idx:..., …}").
    3. ``adapter.system_prompt`` is prepended as the system role.
    4. ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
       wraps both into the Qwen ``<|im_start|>system…<|im_start|>user…
       <|im_start|>assistant`` format so the instruct-tuned base model
       knows to respond as the assistant.

    The result is the prompt string the policy will actually see at
    generation time; the JSONL's own ``prompt``/``completion`` fields
    are intentionally **ignored** (they're for offline reward-model
    regression, not GRPO).

    Raises
    ------
    FileNotFoundError
        If the JSONL path doesn't exist.
    ValueError
        If no rows match the requested ``env_id`` (typo-catch).
    """
    from datasets import Dataset  # noqa: PLC0415
    from verifiable_labs_envs.reward_distillation.dataset import (  # noqa: PLC0415
        read_jsonl,
    )

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset JSONL not found: {path}")

    env, adapter = _load_env_and_adapter(env_id)
    system_msg: str = getattr(adapter, "system_prompt", "") or ""

    seeds_seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if row.env_id != env_id:
            continue
        md = row.metadata if isinstance(row.metadata, dict) else {}
        seed_val = md.get("seed")
        if seed_val is None:
            continue
        seed_int = int(seed_val)
        if seed_int in seeds_seen:
            # Bit-stable dedupe: a single (env_id, seed) pair only
            # contributes once to the GRPO prompt set even if the
            # JSONL has duplicates across sources / runs.
            continue
        seeds_seen.add(seed_int)

        instance = env.generate_instance(seed=seed_int)
        user_msg = adapter.build_user_prompt(instance)
        chat_messages: list[dict[str, str]] = []
        if system_msg:
            chat_messages.append({"role": "system", "content": system_msg})
        chat_messages.append({"role": "user", "content": user_msg})

        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "prompt": prompt,
                seed_kwarg: seed_int,
            }
        )

    if not rows:
        raise ValueError(
            f"no rows with env_id={env_id!r} found in {path}; "
            "check the dataset or override --env-id"
        )

    _LOG.info(
        "loaded %d GRPO prompts for env_id=%s (rebuilt via adapter+chat-template) "
        "from %s",
        len(rows),
        env_id,
        path,
    )
    return Dataset.from_list(rows)


def build_grpo_trainer(config: TrainingConfig) -> Any:
    """Construct a TRL 1.4 ``GRPOTrainer`` wired to:

    * the env-side reward function from
      :mod:`verifiable_labs_envs.training.reward_fn` for
      ``config.env_id``;
    * the LoRA adapter spec from :mod:`vlabs_reward_train.lora_config`;
    * an HF :class:`datasets.Dataset` of prompts rebuilt via the env's
      LLM adapter + the tokenizer's chat template (see
      :func:`_load_grpo_prompts_dataset`);
    * colocate vLLM 0.21 for fast generation (the soft TRL/vLLM
      version-mismatch warning is non-fatal — verified end-to-end
      under torch 2.11.0+cu130 on Blackwell sm120).

    The returned trainer is ready to call ``.train()``. The CLI's
    ``train`` command does exactly that.

    Notes
    -----
    * ``VLLM_USE_FLASHINFER_SAMPLER`` is forced to ``"0"`` because the
      stock FlashInfer sampler has a stale sm75-only check that
      raises on Blackwell sm120 (workaround inherited from the
      Phase 18 smoke fix, commit 8984e0b).
    * ``LD_LIBRARY_PATH`` is intentionally **not** touched — callers
      are expected to set it to include
      ``site-packages/nvidia/cu13/lib`` when launching from a shell
      where ``libcudart.so.13`` isn't otherwise discoverable.
    * Tokenizer + model are loaded under
      ``Qwen/Qwen2.5-1.5B-Instruct`` from the HF cache; if the cache
      is cold (e.g. fresh pod) the model is downloaded once (~30 s
      on RunPod's network).
    """
    # FlashInfer sm75 check workaround for Blackwell sm120 (see Phase 18).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Lazy imports — keep CPU-only paths (dry-run, CI) fast.
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from vlabs_reward_train.lora_config import build_peft_lora_config  # noqa: PLC0415
    from verifiable_labs_envs.training.reward_fn import make_reward_fn  # noqa: PLC0415

    train_kwargs = build_training_args(config)

    grpo_cfg = GRPOConfig(
        use_vllm=True,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        **train_kwargs,
    )

    _LOG.info("loading tokenizer for %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    _LOG.info(
        "loading prompts dataset (env_id=%s) from %s",
        config.env_id,
        config.dataset_path,
    )
    train_ds = _load_grpo_prompts_dataset(
        config.dataset_path,
        env_id=config.env_id,
        tokenizer=tokenizer,
    )

    _LOG.info("building reward fn for env_id=%s", config.env_id)
    reward_fn = make_reward_fn(config.env_id)

    _LOG.info("building LoRA config %s", config.lora_spec)
    lora_cfg = build_peft_lora_config(config.lora_spec)

    _LOG.info(
        "constructing GRPOTrainer (model=%s, batch=%d, num_generations=%d, "
        "max_steps=%d, vllm_gpu_util=%.2f)",
        config.base_model,
        config.batch_size,
        config.num_generations,
        config.max_steps,
        config.vllm_gpu_memory_utilization,
    )
    trainer = GRPOTrainer(
        model=config.base_model,
        args=grpo_cfg,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )
    return trainer


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

    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_BF16",
    "DEFAULT_ENV_ID",
    "DEFAULT_EPOCHS",
    "DEFAULT_GRAD_ACCUM",
    "DEFAULT_LR",
    "DEFAULT_MAX_COMPLETION_LENGTH",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_NUM_GENERATIONS",
    "DEFAULT_VLLM_GPU_MEMORY_UTILIZATION",
    "DEFAULT_VLLM_MAX_MODEL_LENGTH",
    "DEFAULT_VLLM_MODE",
    "DEFAULT_VLLM_TENSOR_PARALLEL_SIZE",
    "REQUIRED_DEPS",
    "DependencyStatus",
    "GpuPathNotImplemented",
    "TrainingConfig",
    "_load_env_and_adapter",
    "_load_grpo_prompts_dataset",
    "build_grpo_trainer",
    "build_training_args",
    "validate_dependencies",
    "write_run_card",
]
