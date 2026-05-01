"""Unit tests for ``vlabs_audit.config``."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlabs_audit.config import AuditConfig, load_config


def test_minimal_yaml_loads(tmp_yaml_config: Path) -> None:
    cfg = load_config(tmp_yaml_config, {})
    assert cfg.model == "claude-haiku-4.5"
    assert cfg.envs == ["sparse-fourier-recovery"]
    assert cfg.episodes == 5
    assert cfg.alpha == 0.1
    assert cfg.parallel == 1  # pydantic default
    assert cfg.anonymize is False  # pydantic default


def test_cli_overrides_yaml(tmp_yaml_config: Path) -> None:
    cfg = load_config(
        tmp_yaml_config,
        {"episodes": 100, "alpha": 0.05, "parallel": 4, "seed_start": 999},
    )
    assert cfg.episodes == 100
    assert cfg.alpha == 0.05
    assert cfg.parallel == 4
    assert cfg.seed_start == 999
    # Untouched values stay from YAML
    assert cfg.model == "claude-haiku-4.5"


def test_none_overrides_ignored(tmp_yaml_config: Path) -> None:
    cfg = load_config(
        tmp_yaml_config, {"episodes": None, "alpha": None, "parallel": None}
    )
    assert cfg.episodes == 5  # YAML wins, None did not clobber


def test_no_yaml_no_overrides_fails() -> None:
    with pytest.raises(ValidationError):
        load_config(None, None)


def test_alpha_bounds() -> None:
    with pytest.raises(ValidationError):
        AuditConfig(model="x", envs=["e"], output=Path("/t.pdf"), alpha=0.0)
    with pytest.raises(ValidationError):
        AuditConfig(model="x", envs=["e"], output=Path("/t.pdf"), alpha=1.0)


def test_episodes_lower_bound() -> None:
    with pytest.raises(ValidationError):
        AuditConfig(model="x", envs=["e"], output=Path("/t.pdf"), episodes=0)


def test_envs_must_be_nonempty() -> None:
    with pytest.raises(ValidationError):
        AuditConfig(model="x", envs=[], output=Path("/t.pdf"))


def test_extra_fields_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "model: x\nenvs: [e]\noutput: /t.pdf\nbogus_field: 42\n"
    )
    with pytest.raises(ValidationError):
        load_config(p, {})
