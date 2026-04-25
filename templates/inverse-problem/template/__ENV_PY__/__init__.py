"""__ENV_ID__ — Verifiable Labs scientific RL environment.

Domain: __DOMAIN__.

This is a scaffolded env generated from
``templates/inverse-problem/`` via
``scripts/create_env.py __ENV_ID__ --domain __DOMAIN__``.

Replace the ``NotImplementedError`` stubs in ``forward_op.py``,
``data.py``, ``reward.py``, ``adapter.py``, and ``env.py`` with your
domain-specific logic, then run
``scripts/validate_env.py environments/__ENV_PY__/`` from the repo
root to verify all four scaffold checks pass.
"""
from __future__ import annotations

__version__ = "0.1.0-alpha"

from __ENV_PY__.env import __ENV_CLASS__, load_environment

ENV_ID = "__ENV_ID__"
DOMAIN = "__DOMAIN__"
# Effective instance count = |seed_space| × |ground_truth_pool|. Used by
# scripts/validate_env.py's procedural-regeneration check, which expects
# > 1e15 unique measurement strings to certify contamination-resistance.
EFFECTIVE_INSTANCES: int = 2**64 * 1024  # TODO: adjust |ground_truth_pool|

__all__ = [
    "ENV_ID",
    "DOMAIN",
    "EFFECTIVE_INSTANCES",
    "__ENV_CLASS__",
    "load_environment",
    "__version__",
]
