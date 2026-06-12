"""__ENV_ID__ — Verifiable Labs SQL RL environment.

Domain: __DOMAIN__.

Scaffolded from ``templates/sql-execution/`` via
``scripts/create_env.py __ENV_ID__ --template sql-execution --domain __DOMAIN__``.

Replace the ``NotImplementedError`` stub in ``data.py`` (and adjust
hyperparameters in ``env.py`` if needed), then run
``scripts/validate_env.py environments/__ENV_PY__/`` from the repo
root to verify the scaffold contract.
"""
from __future__ import annotations

__version__ = "0.1.0-alpha"

from __ENV_PY__.env import __ENV_CLASS__, load_environment

ENV_ID = "__ENV_ID__"
DOMAIN = "__DOMAIN__"
# Effective instance count = |seed_space| × |template_pool| ×
# |per-template parameter combinations|. The validator's
# procedural-regeneration check requires > 1e15 unique instances.
# Default: 64-bit seed × 8 templates × ~1e5 combinations ≈ 1.5e23.
EFFECTIVE_INSTANCES: int = 8 * (2**64) * 1_000_000

__all__ = [
    "ENV_ID",
    "DOMAIN",
    "EFFECTIVE_INSTANCES",
    "__ENV_CLASS__",
    "load_environment",
    "__version__",
]
