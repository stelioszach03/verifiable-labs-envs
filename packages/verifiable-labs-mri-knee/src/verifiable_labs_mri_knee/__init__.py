"""Verifiable Labs env wrapper: mri-knee-reconstruction.

Thin re-export over the monorepo at https://github.com/stelioszach03/verifiable-labs-envs.
"""
from verifiable_labs_envs.envs.mri_knee import (
    load_environment as _load_environment_base,
)
from verifiable_labs_envs.envs.mri_knee_multiturn import (
    load_environment as _load_environment_multiturn_base,
)

ENV_NAME = "mri-knee-reconstruction"
MT_ENV_NAME = "mri-knee-reconstruction-multiturn"
__version__ = "1.0.0"


def load_environment(*args, **kwargs):
    return _load_environment_base(*args, **kwargs)


def load_environment_multiturn(*args, **kwargs):
    return _load_environment_multiturn_base(*args, **kwargs)


__all__ = [
    "ENV_NAME",
    "MT_ENV_NAME",
    "load_environment",
    "load_environment_multiturn",
    "__version__",
]
