"""Multi-turn tool-use with primitive composition.

Prime Intellect Hub wrapper around ``verifiable_labs_envs.envs.tool_calling_multiturn``.
The monorepo at https://github.com/stelioszach03/verifiable-labs-envs is the
source of truth; this file is a thin re-export so the env can be installed
and discovered via the Prime Intellect Environments Hub.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.tool_calling_multiturn import load_environment as _le


def load_environment(**kwargs: Any):
    """Factory for the ``tool-calling-multiturn`` environment."""
    return _le(**kwargs)


__all__ = ["load_environment"]
