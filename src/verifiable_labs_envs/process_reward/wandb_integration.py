"""W&B integration for the PRM trainer (Phase 30.C scaffold).

Reuses Phase 29's
:func:`vlabs_reward_train.wandb_callback.init_wandb_run` for the
run handle + lifecycle helpers. PRM-specific additions:

- :func:`log_per_step_metrics` — pushes a per-step-loss decomposition
  to W&B at each eval pass, plus the moat-aligned metrics
  (`per_step_calibration_coverage`, `bon_lift_vs_phase29`).
- :func:`log_multi_task_balance` — when D4-D multi-task is enabled,
  logs the relative magnitudes of the per-step vs outcome losses so
  ablation runs can spot the joint training balance.
- :func:`log_run_card` — flattens a PRM training-config + dependency
  status into a W&B run config payload at training start.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from vlabs_reward_train.wandb_callback import (
    DEFAULT_MODE,
    WandbHandle,
    has_wandb_credentials,
    init_wandb_run,
    is_wandb_available,
    log_metrics,
    wandb_run,
)

logger = logging.getLogger(__name__)

DEFAULT_PRM_PROJECT: str = "vlabs-prm-distillation"


def init_prm_run(
    *,
    project: str = DEFAULT_PRM_PROJECT,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    mode: str = DEFAULT_MODE,
    fallback_to_noop: bool = True,
) -> WandbHandle:
    """Initialise a W&B run for the PRM training loop.

    Thin wrapper around :func:`init_wandb_run` that locks the default
    project to ``vlabs-prm-distillation`` (vs Phase 29's
    ``vlabs-reward-distillation``). All other lifecycle semantics
    inherit from Phase 29.
    """
    return init_wandb_run(
        project=project,
        name=name,
        config=dict(config or {}),
        mode=mode,
        fallback_to_noop=fallback_to_noop,
    )


def log_per_step_metrics(
    handle: WandbHandle,
    step: int,
    *,
    per_step_loss: float,
    per_step_calibration_coverage: float | None = None,
    per_step_calibration_drift: float | None = None,
    bon_lift_vs_phase29: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Log the PRM-specific metric set at one training/eval step.

    Always-on fields:

    - ``prm/per_step_loss`` — the masked-MSE per-step loss.

    Optional fields (logged when supplied):

    - ``prm/per_step_calibration_coverage`` — D9-C empirical coverage
      averaged across step positions.
    - ``prm/per_step_calibration_drift`` — coverage minus target
      (D9-C: should be within ±5pp).
    - ``prm/bon_lift_vs_phase29`` — D6-B headline (BoN reranking lift
      over the Phase 29 distilled RM baseline).

    All metrics are tagged with ``step`` so the W&B X-axis lines up
    with the trainer's global-step counter.
    """
    payload: dict[str, Any] = {"prm/per_step_loss": float(per_step_loss)}
    if per_step_calibration_coverage is not None:
        payload["prm/per_step_calibration_coverage"] = float(
            per_step_calibration_coverage
        )
    if per_step_calibration_drift is not None:
        payload["prm/per_step_calibration_drift"] = float(
            per_step_calibration_drift
        )
    if bon_lift_vs_phase29 is not None:
        payload["prm/bon_lift_vs_phase29"] = float(bon_lift_vs_phase29)
    if extra:
        for k, v in extra.items():
            payload[f"prm/{k}"] = v
    log_metrics(handle, step=int(step), metrics=payload)


def log_multi_task_balance(
    handle: WandbHandle,
    step: int,
    *,
    step_loss_component: float,
    outcome_loss_component: float,
    per_step_weight: float,
    outcome_weight: float,
) -> None:
    """Push the per-step vs outcome loss-magnitude ratio + the active
    blend weights, so the ablation dashboard catches drift in either
    direction.
    """
    total = float(step_loss_component) + float(outcome_loss_component)
    payload = {
        "prm/multi_task/step_loss_component": float(step_loss_component),
        "prm/multi_task/outcome_loss_component": float(outcome_loss_component),
        "prm/multi_task/per_step_weight": float(per_step_weight),
        "prm/multi_task/outcome_weight": float(outcome_weight),
        "prm/multi_task/total_loss": total,
        "prm/multi_task/step_share": (
            float(step_loss_component) / total if total > 0 else 0.0
        ),
    }
    log_metrics(handle, step=int(step), metrics=payload)


def log_run_card(
    handle: WandbHandle,
    *,
    config: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    multi_task: Mapping[str, Any],
) -> None:
    """Flatten the training config into the W&B run-config payload.

    Called once at training start. Safe to invoke on the no-op
    handle.
    """
    if not handle.is_real:
        return
    flattened: dict[str, Any] = {f"config/{k}": v for k, v in config.items()}
    flattened.update({f"deps/{k}": v for k, v in dependencies.items()})
    flattened.update({f"multi_task/{k}": v for k, v in multi_task.items()})
    handle.log(flattened)


def aggregate_step_loss_decomposition(
    per_step_losses: Sequence[float],
) -> dict[str, float]:
    """Compute mean / max / position-of-max diagnostics from a list of
    per-step loss values. Used by :func:`log_per_step_metrics` callers
    when the trainer exposes per-position loss arrays."""
    if not per_step_losses:
        return {"mean": 0.0, "max": 0.0, "argmax": 0.0, "n": 0.0}
    losses = [float(x) for x in per_step_losses]
    max_value = max(losses)
    argmax = float(losses.index(max_value))
    return {
        "mean": sum(losses) / len(losses),
        "max": max_value,
        "argmax": argmax,
        "n": float(len(losses)),
    }


__all__ = [
    "DEFAULT_PRM_PROJECT",
    "WandbHandle",
    "aggregate_step_loss_decomposition",
    "has_wandb_credentials",
    "init_prm_run",
    "is_wandb_available",
    "log_multi_task_balance",
    "log_per_step_metrics",
    "log_run_card",
    "wandb_run",
]
