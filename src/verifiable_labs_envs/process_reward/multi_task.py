"""D4-D multi-task config + D13-C hybrid path scaffold (Phase 30.C).

The PRM student (Qwen2.5-1.5B + LoRA) optionally co-trains an
**outcome head** alongside the per-step regression head. The
outcome head matches the Phase 29 distilled RM head shape; under
:doc:`PHASE_30_PLAN.md` D13-B the two heads share the same backbone
and the outcome head is *literally* the Phase 29 RM head fine-tuned
jointly. Under D13-C (the locked v0.0.1 hybrid path), 30.C ships
the multi-task scaffolding but defaults to D13-A (independent
serving) — operators flip the toggle in 30.F when the joint training
proves stable.

Public surface:

- :class:`MultiTaskConfig` — frozen dataclass holding the loss
  weights + head wiring options.
- :func:`build_multi_task_loss` — returns a callable producing the
  joint scalar loss from per-step preds + outcome pred + targets.
- :func:`split_phase29_compat_payload` — extracts the outcome target
  + components from a :class:`ProcessRewardTraceRow` so the joint
  trainer can feed both heads from a single dataset.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verifiable_labs_envs.process_reward.dataset import ProcessRewardTraceRow

DEFAULT_OUTCOME_WEIGHT: float = 0.3
"""D4-D blend ratio per :doc:`PHASE_30_PLAN.md` §5."""

DEFAULT_PER_STEP_WEIGHT: float = 1.0 - DEFAULT_OUTCOME_WEIGHT
"""Always equals ``1 - outcome_weight``."""


@dataclass(frozen=True)
class MultiTaskConfig:
    """Joint per-step + outcome loss configuration.

    Fields:

    - ``per_step_weight`` / ``outcome_weight`` — non-negative loss
      weights (``per_step_weight + outcome_weight`` need not sum to 1
      in general; the trainer's outer scheduler can rescale either
      independently).
    - ``share_backbone`` — D13-B/C path. When ``True`` the trainer
      loads + fine-tunes the Phase 29 RM LoRA adapters as the
      starting point; when ``False`` the trainer initialises fresh
      LoRA adapters (D13-A independent serving).
    - ``freeze_outcome_head`` — when ``True`` (D13-C v0.0.2 path),
      the outcome head's parameters are frozen during PRM training so
      the joint loss only updates the per-step head + shared backbone
      adapters. Acts as a regulariser on the per-step head and
      eliminates R16 (outcome head regression). Default ``False``
      enables full joint co-training (D13-B path).
    """

    per_step_weight: float = DEFAULT_PER_STEP_WEIGHT
    outcome_weight: float = DEFAULT_OUTCOME_WEIGHT
    share_backbone: bool = False
    freeze_outcome_head: bool = False
    enable: bool = False
    """Master toggle. When ``False``, multi-task is OFF — the trainer
    only optimises the per-step head and ``build_multi_task_loss``
    returns a per-step-only loss callable."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_step_weight": float(self.per_step_weight),
            "outcome_weight": float(self.outcome_weight),
            "share_backbone": bool(self.share_backbone),
            "freeze_outcome_head": bool(self.freeze_outcome_head),
            "enable": bool(self.enable),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiTaskConfig:
        return cls(
            per_step_weight=float(
                payload.get("per_step_weight", DEFAULT_PER_STEP_WEIGHT)
            ),
            outcome_weight=float(
                payload.get("outcome_weight", DEFAULT_OUTCOME_WEIGHT)
            ),
            share_backbone=bool(payload.get("share_backbone", False)),
            freeze_outcome_head=bool(payload.get("freeze_outcome_head", False)),
            enable=bool(payload.get("enable", False)),
        )

    def normalised(self) -> MultiTaskConfig:
        """Return a copy with ``per_step_weight + outcome_weight = 1``."""
        total = self.per_step_weight + self.outcome_weight
        if total <= 0:
            raise ValueError(
                "per_step_weight + outcome_weight must be > 0; "
                f"got {self.per_step_weight} + {self.outcome_weight}"
            )
        return MultiTaskConfig(
            per_step_weight=self.per_step_weight / total,
            outcome_weight=self.outcome_weight / total,
            share_backbone=self.share_backbone,
            freeze_outcome_head=self.freeze_outcome_head,
            enable=self.enable,
        )


# ── loss builder (CPU-friendly torch path) ──────────────────────────


def build_multi_task_loss(
    config: MultiTaskConfig,
) -> Callable[[Any, Any, Any, Any, Any | None], Any]:
    """Return a callable that computes the joint scalar loss.

    The callable signature is::

        loss_fn(per_step_preds, per_step_targets, per_step_mask,
                outcome_preds=None, outcome_targets=None) -> scalar Tensor

    where ``per_step_preds`` / ``per_step_targets`` are
    ``(batch, max_steps)`` tensors, ``per_step_mask`` is a boolean
    mask over valid step positions, ``outcome_preds`` /
    ``outcome_targets`` are ``(batch,)`` tensors required only when
    :attr:`MultiTaskConfig.enable` is True.

    Loss formula:

    ```
    L_step = sum_b sum_t mask[b,t] * (preds[b,t] - targets[b,t])^2
             / max(1, sum mask)
    L_outcome = mean((outcome_preds - outcome_targets)^2)
    L_total = per_step_weight * L_step
              + (outcome_weight * L_outcome  if enable else 0)
    ```

    Lazy torch import — the loss callable can be constructed in
    CPU-only paths (the default 30.C test harness) and only touches
    torch when invoked on real tensors.
    """
    import torch  # noqa: PLC0415

    def loss_fn(
        per_step_preds: torch.Tensor,
        per_step_targets: torch.Tensor,
        per_step_mask: torch.Tensor,
        outcome_preds: torch.Tensor | None = None,
        outcome_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_f = per_step_mask.to(per_step_preds.dtype)
        valid = mask_f.sum().clamp_min(1.0)
        squared = (per_step_preds - per_step_targets) ** 2
        l_step = (squared * mask_f).sum() / valid

        if not config.enable:
            return config.per_step_weight * l_step

        if outcome_preds is None or outcome_targets is None:
            raise ValueError(
                "outcome_preds and outcome_targets must be supplied "
                "when MultiTaskConfig.enable is True"
            )
        l_outcome = ((outcome_preds - outcome_targets) ** 2).mean()
        return config.per_step_weight * l_step + config.outcome_weight * l_outcome

    return loss_fn


# ── dataset helpers for joint training ──────────────────────────────


def split_phase29_compat_payload(
    rows: Sequence[ProcessRewardTraceRow],
) -> dict[str, Any]:
    """Build the multi-task batch dict from a list of trace rows.

    Returns a dict with:

    - ``per_step_targets`` / ``per_step_mask`` — packed per-step
      consensus rewards + valid-position mask
      (see :func:`verifiable_labs_envs.process_reward.trainer.per_step_target_tensor`).
    - ``outcome_targets`` — ``(batch,)`` tensor of aggregate rewards.
    - ``prompts`` / ``traces_joined`` — the text inputs for the
      tokenizer (a flat list of strings each).

    Lazy torch import inside the helper.
    """
    from verifiable_labs_envs.process_reward.trainer import (
        per_step_outcome_tensor,
        per_step_target_tensor,
    )

    if not rows:
        return {
            "per_step_targets": per_step_target_tensor([])["targets"],
            "per_step_mask": per_step_target_tensor([])["mask"],
            "outcome_targets": per_step_outcome_tensor([]),
            "prompts": [],
            "traces_joined": [],
        }

    targets_dict = per_step_target_tensor(rows)
    return {
        "per_step_targets": targets_dict["targets"],
        "per_step_mask": targets_dict["mask"],
        "outcome_targets": per_step_outcome_tensor(rows),
        "prompts": [r.prompt for r in rows],
        "traces_joined": ["\n".join(r.steps) for r in rows],
    }


# ── D13-C hybrid path helpers ───────────────────────────────────────


def is_shared_backbone_ready(base_rm_checkpoint: str | None) -> bool:
    """Predicate: is the D13-B/C path activatable for this run?

    Returns ``True`` iff a Phase 29 RM checkpoint is supplied.
    Used by the trainer dry-run + the CLI to surface which path is
    selected without loading anything.
    """
    return base_rm_checkpoint is not None and bool(base_rm_checkpoint)


def loss_summary(config: MultiTaskConfig) -> dict[str, Any]:
    """JSON-serialisable summary of the loss configuration.

    Used by the W&B run-config payload + the run_card.json so the
    audit consumer can answer "did this run train both heads?" at a
    glance.
    """
    norm = config.normalised() if (config.per_step_weight + config.outcome_weight) > 0 else config
    return {
        "enable": config.enable,
        "per_step_weight": config.per_step_weight,
        "outcome_weight": config.outcome_weight,
        "per_step_weight_normalised": (
            norm.per_step_weight if config.enable else 1.0
        ),
        "outcome_weight_normalised": (
            norm.outcome_weight if config.enable else 0.0
        ),
        "share_backbone": config.share_backbone,
        "freeze_outcome_head": config.freeze_outcome_head,
    }


__all__ = [
    "DEFAULT_OUTCOME_WEIGHT",
    "DEFAULT_PER_STEP_WEIGHT",
    "MultiTaskConfig",
    "build_multi_task_loss",
    "is_shared_backbone_ready",
    "loss_summary",
    "split_phase29_compat_payload",
]
