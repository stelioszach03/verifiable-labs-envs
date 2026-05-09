"""W&B integration with offline-mode fallback (Phase 29.C scaffold).

Per :doc:`PHASE_29_PLAN.md` §8 smoke-test contract: the W&B integration
must use ``wandb.init(mode="offline")`` so CI doesn't need a real W&B
account. The 29.F production runs flip ``mode="online"`` and pass a
real ``WANDB_API_KEY``.

This module is the thin shim that abstracts over the two modes:

- :func:`init_wandb_run` returns a context-manager-shaped wrapper that
  cleanly handles the ``wandb`` library being absent (returns a no-op
  callback so the trainer's outer loop doesn't branch).
- :func:`log_metrics` / :func:`log_calibration_card` are thin wrappers
  that the trainer calls at each eval step.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROJECT: str = "vlabs-reward-distillation"
DEFAULT_MODE: str = "offline"


@dataclass
class WandbHandle:
    """Lightweight wrapper around a wandb run + the no-op fallback.

    ``run`` is the real ``wandb.sdk.wandb_run.Run`` (typed as ``Any``
    here so the module imports without wandb installed) when the real
    library is available; ``None`` for the no-op fallback path.
    """

    run: Any = None
    mode: str = DEFAULT_MODE
    project: str = DEFAULT_PROJECT

    @property
    def is_real(self) -> bool:
        """True when an actual wandb run is attached."""
        return self.run is not None

    def log(self, payload: Mapping[str, Any]) -> None:
        """Push a metrics dict to wandb (or noop)."""
        if not self.is_real:
            return
        self.run.log(dict(payload))

    def finish(self) -> None:
        """Close the run cleanly. Safe to call on the no-op handle."""
        if not self.is_real:
            return
        try:
            self.run.finish()
        except Exception as exc:  # noqa: BLE001 — wandb finish errors shouldn't kill the trainer
            logger.warning("wandb finish failed: %s", exc)


def init_wandb_run(
    *,
    project: str = DEFAULT_PROJECT,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    mode: str = DEFAULT_MODE,
    fallback_to_noop: bool = True,
) -> WandbHandle:
    """Initialise a W&B run, falling back to a no-op handle when wandb
    is unavailable or ``mode`` is "disabled".

    Defaults to ``mode="offline"`` so CI / headless runs work without a
    network. Pass ``mode="online"`` plus a ``WANDB_API_KEY`` env var
    in 29.F production runs.
    """
    if mode == "disabled":
        return WandbHandle(run=None, mode=mode, project=project)

    try:
        import wandb  # noqa: PLC0415 — lazy
    except ImportError as exc:
        if not fallback_to_noop:
            raise RuntimeError("wandb not installed; pass fallback_to_noop=True") from exc
        logger.info("wandb not installed; using no-op handle")
        return WandbHandle(run=None, mode=mode, project=project)

    try:
        run = wandb.init(  # type: ignore[attr-defined]
            project=project,
            name=name,
            config=dict(config or {}),
            mode=mode,
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        if not fallback_to_noop:
            raise RuntimeError(f"wandb.init failed: {exc}") from exc
        logger.warning("wandb.init failed (%s); falling back to no-op", exc)
        return WandbHandle(run=None, mode=mode, project=project)
    return WandbHandle(run=run, mode=mode, project=project)


@contextmanager
def wandb_run(
    *,
    project: str = DEFAULT_PROJECT,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    mode: str = DEFAULT_MODE,
) -> Iterator[WandbHandle]:
    """Context-managed wandb run; finishes cleanly on scope exit."""
    handle = init_wandb_run(project=project, name=name, config=config, mode=mode)
    try:
        yield handle
    finally:
        handle.finish()


def log_metrics(handle: WandbHandle, step: int, metrics: Mapping[str, Any]) -> None:
    """Log a metrics dict tagged with ``step``. Safe on a no-op handle."""
    payload = {"step": int(step), **dict(metrics)}
    handle.log(payload)


def log_calibration_card(handle: WandbHandle, card: Mapping[str, Any]) -> None:
    """Push the calibration card (D10 quantile + drift) to W&B as a
    metrics dict. The card is always logged at step 0 of the eval pass
    so dashboard panels render the calibration trace cleanly."""
    if not handle.is_real:
        return
    flattened = {f"calibration/{k}": v for k, v in dict(card).items()}
    handle.log(flattened)


def is_wandb_available() -> bool:
    """Predicate: is the ``wandb`` library importable?"""
    try:
        import wandb  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def has_wandb_credentials() -> bool:
    """Predicate: is ``WANDB_API_KEY`` set? Used by the CLI's
    ``dependencies`` command to surface friendly setup messages."""
    return bool(os.environ.get("WANDB_API_KEY", "").strip())


__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_PROJECT",
    "WandbHandle",
    "has_wandb_credentials",
    "init_wandb_run",
    "is_wandb_available",
    "log_calibration_card",
    "log_metrics",
    "wandb_run",
]
