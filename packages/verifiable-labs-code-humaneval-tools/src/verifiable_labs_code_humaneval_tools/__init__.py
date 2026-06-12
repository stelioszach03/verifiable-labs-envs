"""Verifiable Labs env wrapper: code-humaneval-tools.

Tool-use procedural code-execution RL environment with
read_file/write_file/run_test primitives. Thin re-export over
``verifiable_labs_envs.envs``.
"""
from verifiable_labs_envs.envs.code_humaneval_tools import (
    load_environment as _load_environment_base,
)

ENV_NAME = "code-humaneval-tools"
__version__ = "0.1.0"


def load_environment(*args, **kwargs):
    """Factory for the ``code-humaneval-tools`` environment."""
    return _load_environment_base(*args, **kwargs)


__all__ = ["ENV_NAME", "load_environment", "__version__"]
