"""Verifiable Labs reward-distillation training scaffolding (Phase 29.C).

Heavy deps (torch / transformers / peft / trl / wandb) are imported
lazily inside the relevant submodules so the package's surface stays
importable on CPU-only environments. The `train` command guards on
:func:`vlabs_reward_train.trainer.validate_dependencies` and refuses
to proceed if anything is missing.
"""
from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
