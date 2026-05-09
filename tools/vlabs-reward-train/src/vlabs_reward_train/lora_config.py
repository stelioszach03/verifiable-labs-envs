"""PEFT LoraConfig defaults for the distilled reward model (D3-A).

Per :doc:`PHASE_29_PLAN.md` §5 D3-A: LoRA rank 16 / alpha 32 applied to
the attention `q_proj`, `k_proj`, `v_proj`, `o_proj` matrices on a
Qwen2.5-1.5B-Instruct base. Trains ~1.6 % of total params, checkpoint
~30 MB.

This module exposes:

- :data:`DEFAULT_LORA_CONFIG` — the locked dict, snake-case keys.
- :func:`build_peft_lora_config` — adapter that wraps the dict in a
  ``peft.LoraConfig`` instance, importing peft *lazily* so the surface
  stays importable in CPU-only/dev environments where peft isn't
  installed.
- :func:`lora_target_param_fraction` — sanity check helper that
  estimates the trainable-fraction; useful for the `dependencies`
  CLI command and ablation logging.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_LORA_R: int = 16
DEFAULT_LORA_ALPHA: int = 32
DEFAULT_LORA_DROPOUT: float = 0.05
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_BIAS: str = "none"
DEFAULT_TASK_TYPE: str = "SEQ_CLS"
"""Sequence-classification task type — the reward head projects the
last hidden state to a scalar in [0, 1]."""


@dataclass(frozen=True)
class LoraSpec:
    """Frozen view of the LoRA hyperparameters.

    All fields default to the locked D3-A values per
    :doc:`PHASE_29_PLAN.md`. Override only via the explicit constructor
    keyword arguments — Phase 29.F ablation sweeps will use this to
    record per-experiment config in the W&B run table.
    """

    r: int = DEFAULT_LORA_R
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    bias: str = DEFAULT_BIAS
    task_type: str = DEFAULT_TASK_TYPE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation; safe to pickle into the
        run-config column of a `dataset_jobs` row."""
        return {
            "r": int(self.r),
            "alpha": int(self.alpha),
            "dropout": float(self.dropout),
            "target_modules": list(self.target_modules),
            "bias": str(self.bias),
            "task_type": str(self.task_type),
            **self.extra,
        }

    def with_overrides(self, **overrides: Any) -> LoraSpec:
        """Return a new spec with the given keyword overrides applied."""
        if not overrides:
            return self
        kwargs = {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "extra": dict(self.extra),
        }
        for k, v in overrides.items():
            if k == "target_modules" and not isinstance(v, tuple):
                v = tuple(v)
            kwargs[k] = v
        return LoraSpec(**kwargs)


DEFAULT_LORA_CONFIG: dict[str, Any] = LoraSpec().to_dict()
"""Locked default LoraConfig dict per :doc:`PHASE_29_PLAN.md` D3-A."""


def build_peft_lora_config(spec: LoraSpec | None = None, **overrides: Any) -> Any:
    """Lazily import ``peft`` and return a ``peft.LoraConfig`` instance.

    Falls back to a :class:`RuntimeError` if peft isn't installed —
    callers should either catch this or guard with
    :func:`vlabs_reward_train.trainer.validate_dependencies` first.
    """
    try:
        import peft  # noqa: PLC0415 — lazy
    except ImportError as exc:
        raise RuntimeError(
            "peft is not installed; `pip install vlabs-reward-train[gpu]` "
            "to pull the GPU-training extras (Phase 29.F path)."
        ) from exc

    final_spec = (spec or LoraSpec()).with_overrides(**overrides)
    return peft.LoraConfig(
        r=final_spec.r,
        lora_alpha=final_spec.alpha,
        lora_dropout=final_spec.dropout,
        target_modules=list(final_spec.target_modules),
        bias=final_spec.bias,
        task_type=final_spec.task_type,
        **final_spec.extra,
    )


def lora_target_param_fraction(
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    *,
    n_layers: int = 28,
    rank: int = DEFAULT_LORA_R,
    hidden_size: int = 1536,
    full_param_count: int = 1_500_000_000,
) -> float:
    """Rough estimate of the *trainable* parameter fraction under D3-A.

    For a single attention layer with ``len(target_modules)`` projections,
    each of shape ``hidden_size × hidden_size``, LoRA replaces the full
    matrix with two ``hidden_size × rank`` matrices (≈ ``2 * rank *
    hidden_size`` trainable params). The total trainable count over
    the whole model is then ``n_layers * len(target_modules) * 2 *
    rank * hidden_size``.

    Defaults match Qwen2.5-1.5B (28 transformer layers, 1536 hidden
    size). The plan §5 D3-A claim "trains ~1.6% of total parameters"
    implies this should land near 0.016 ± a hair — the exact fraction
    depends on whether MLP projections are also targeted (we don't,
    by default).
    """
    if n_layers <= 0 or rank <= 0 or hidden_size <= 0 or full_param_count <= 0:
        raise ValueError(
            f"all dimensions must be positive; got "
            f"layers={n_layers}, rank={rank}, hidden={hidden_size}, total={full_param_count}"
        )
    trainable = n_layers * len(target_modules) * 2 * rank * hidden_size
    return float(trainable) / float(full_param_count)


def lora_summary(spec: LoraSpec | None = None) -> dict[str, Any]:
    """Return a JSON-friendly summary of the LoRA spec + estimated
    trainable fraction. Used by the CLI's ``dry-run`` and the W&B
    callback's run-config payload."""
    s = spec or LoraSpec()
    return {
        "spec": s.to_dict(),
        "estimated_trainable_fraction": lora_target_param_fraction(
            target_modules=s.target_modules, rank=s.r
        ),
    }


__all__ = [
    "DEFAULT_BIAS",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_CONFIG",
    "DEFAULT_LORA_DROPOUT",
    "DEFAULT_LORA_R",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_TASK_TYPE",
    "LoraSpec",
    "build_peft_lora_config",
    "lora_summary",
    "lora_target_param_fraction",
]
