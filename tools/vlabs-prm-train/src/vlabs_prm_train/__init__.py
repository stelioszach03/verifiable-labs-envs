"""Verifiable Labs process-reward training scaffolding (Phase 30.C).

Heavy deps (torch / transformers / peft / trl / wandb) are imported
lazily inside the relevant submodules so the package's surface stays
importable on CPU-only environments. The `train` command guards on
:func:`verifiable_labs_envs.process_reward.trainer.validate_dependencies`
and refuses to proceed if anything is missing.
"""
from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
