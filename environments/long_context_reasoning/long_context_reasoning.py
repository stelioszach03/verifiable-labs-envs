"""Multi-hop reasoning over long documents (>10k tokens).

Prime Intellect Hub wrapper around ``verifiable_labs_envs.envs.long_context_reasoning``.
The monorepo at https://github.com/verifiablelabs/verifiable-labs-envs is the
source of truth; this file is a thin re-export so the env can be installed
and discovered via the Prime Intellect Environments Hub.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.long_context_reasoning import load_environment as _le


def load_environment(**kwargs: Any):
    """Factory for the ``long-context-reasoning`` environment."""
    return _le(**kwargs)


__all__ = ["load_environment"]
