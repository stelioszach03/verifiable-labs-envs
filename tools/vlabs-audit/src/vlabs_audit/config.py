"""Pydantic ``AuditConfig`` plus YAML/CLI merge logic.

The CLI accepts both a ``--config FILE.yaml`` and individual flags
(``--model``, ``--episodes``, ...). Flag values override YAML defaults;
``None`` values from the CLI (i.e. flag not provided) leave the YAML
value alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AuditConfig(BaseModel):
    """Validated audit configuration."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="Model id, e.g. 'claude-haiku-4.5'.")
    envs: list[str] = Field(min_length=1, description="Verifiable Labs env ids.")
    episodes: int = Field(default=30, ge=1, le=10_000, description="Episodes per env.")
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0, description="Conformal alpha.")
    output: Path = Field(description="Path to the rendered PDF report.")
    parallel: int = Field(default=1, ge=1, le=16, description="Parallel workers.")
    seed_start: int = Field(default=0, ge=0, description="Starting seed for instances.")
    anonymize: bool = Field(default=False, description="Redact model name in report.")
    anonymize_label: str = Field(
        default="Frontier Model A",
        min_length=1,
        max_length=64,
        description="Label substituted for the model when anonymize=True.",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def load_config(
    config_path: Path | None,
    overrides: dict[str, Any] | None = None,
) -> AuditConfig:
    """Merge YAML config + CLI overrides, return a validated ``AuditConfig``.

    Precedence: ``overrides`` > YAML > pydantic defaults. ``None`` values
    in ``overrides`` are dropped before merging so that a missing CLI flag
    does not clobber a YAML default.
    """
    base: dict[str, Any] = {}
    if config_path is not None:
        base = _load_yaml(config_path)
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
    return AuditConfig.model_validate(base)


__all__ = ["AuditConfig", "load_config"]
