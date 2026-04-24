"""3-turn MRI knee reconstruction with k-space-residual feedback between turns.

Prime Intellect Hub wrapper around ``verifiable_labs_envs.envs.mri_knee_multiturn``.
The monorepo at https://github.com/stelioszach03/verifiable-labs-envs is the
source of truth; this file is a thin re-export so the env can be installed
and discovered via the Prime Intellect Environments Hub.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.mri_knee_multiturn import load_environment as _le


def load_environment(**kwargs: Any):
    """Factory for the ``mri-knee-reconstruction-multiturn`` environment.

    Passes kwargs through to the monorepo's ``load_environment`` (accepts
    ``calibration_quantile``, ``fast``, and env-specific options like
    ``max_turns`` or ``use_real_data`` where applicable).
    """
    return _le(**kwargs)
