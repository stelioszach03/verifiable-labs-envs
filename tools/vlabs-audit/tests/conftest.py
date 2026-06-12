"""Pytest fixtures for vlabs-audit tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_yaml_config(tmp_path: Path) -> Path:
    """A minimal-but-valid YAML config for the merge logic + CLI tests."""
    p = tmp_path / "audit.yaml"
    p.write_text(
        "model: claude-haiku-4.5\n"
        "envs:\n"
        "  - sparse-fourier-recovery\n"
        "episodes: 5\n"
        "alpha: 0.1\n"
        "output: /tmp/x.pdf\n"
    )
    return p
