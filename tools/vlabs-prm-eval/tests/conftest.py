"""Pytest collection-time bootstrap for the vlabs-prm-eval tool tests.

Mirrors the repo-root ``conftest.py``: ensure the repository ``src/`` (the
``verifiable_labs_envs`` package, including ``formal_spec``) and the tool's own
``src/`` are importable, even when these tests are collected with a rootdir of
``tools/vlabs-prm-eval`` rather than the repository root.

Without this, an isolated ``pytest tools/vlabs-prm-eval/tests`` run resolves
``verifiable_labs_envs`` via a stale editable ``.pth`` entry pointing at another
worktree, which may lag this branch's ``formal_spec`` modules. Prepending the
in-tree ``src/`` keeps the tool tests reproducible regardless of rootdir.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tools/vlabs-prm-eval/tests/conftest.py -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = _REPO_ROOT / "src"
_TOOL_SRC = Path(__file__).resolve().parents[1] / "src"

for _p in (_REPO_SRC, _TOOL_SRC):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
