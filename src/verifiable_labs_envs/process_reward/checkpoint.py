"""Checkpoint persistence for the trained PRM (Phase 30.C scaffold).

Per :doc:`PHASE_30_PLAN.md` D12-B + R13: each checkpoint is identified
by a semver `model_id` and persisted both locally + uploaded to R2
(``r2://vlabs-models/{model_id}/{version}/checkpoint/``).

The pattern + manifest shape mirror the Phase 29
:mod:`vlabs_reward_train.checkpointing` module verbatim — only the
manifest field set differs (PRM-specific D9-C per-step quantiles +
D13 base_rm_id link).

Manifest extras vs Phase 29:

- ``base_rm_id`` — when the PRM uses the shared-backbone path
  (D13-B/C), this is the Phase 29 RM ``model_id`` it was initialised
  from.
- ``step_conformal_quantiles`` — JSONB dict of per-step-position
  bucket → quantile (D9-C).
- ``aggregate_conformal_quantile`` — scalar trace-level quantile
  (D9-C).
- ``step_granularity`` — per :doc:`PHASE_30_PLAN.md` D1; locked at
  ``"per_step"`` in v0.0.1.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME: str = "manifest.json"
LORA_WEIGHTS_FILENAME: str = "adapter_model.safetensors"
TRAINING_CONFIG_FILENAME: str = "training_config.json"
SCHEMA_VERSION: str = "v0.1.0"


@dataclass(frozen=True)
class PrmCheckpointManifest:
    """Canonical record of one trained PRM checkpoint.

    Persisted under ``{output_dir}/manifest.json``; uploaded to R2
    next to the LoRA weights.
    """

    model_id: str
    version: str
    base_model: str
    step_granularity: str
    base_rm_id: str | None
    lora_config: dict[str, Any]
    training_config: dict[str, Any]
    multi_task: dict[str, Any]
    metrics: dict[str, Any]
    checkpoint_files: tuple[str, ...]
    step_conformal_quantiles: dict[str, float] | None = None
    aggregate_conformal_quantile: float | None = None
    created_at_unix: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["checkpoint_files"] = list(self.checkpoint_files)
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrmCheckpointManifest:
        return cls(
            model_id=str(payload["model_id"]),
            version=str(payload["version"]),
            base_model=str(payload["base_model"]),
            step_granularity=str(payload.get("step_granularity", "per_step")),
            base_rm_id=payload.get("base_rm_id"),
            lora_config=dict(payload.get("lora_config", {})),
            training_config=dict(payload.get("training_config", {})),
            multi_task=dict(payload.get("multi_task", {})),
            metrics=dict(payload.get("metrics", {})),
            checkpoint_files=tuple(payload.get("checkpoint_files", ())),
            step_conformal_quantiles=(
                dict(payload["step_conformal_quantiles"])
                if payload.get("step_conformal_quantiles")
                else None
            ),
            aggregate_conformal_quantile=payload.get(
                "aggregate_conformal_quantile"
            ),
            created_at_unix=float(payload.get("created_at_unix", time.time())),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint over the manifest (excluding the
        timestamp). Used as the audit-trail row id for trained PRM
        checkpoints in 30.G."""
        payload = {k: v for k, v in self.to_dict().items() if k != "created_at_unix"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()


def model_id_for(family: str, version: str) -> str:
    """Format the locked D12-B PRM model id shape:
    ``vlabs-prm-{family}-v{semver}``.

    Example: ``vlabs-prm-distilled-qwen-1-5b-v0.1.0``.
    """
    if not family:
        raise ValueError("family must be non-empty")
    if not version:
        raise ValueError("version must be non-empty")
    return f"vlabs-prm-{family}-v{version}"


def write_manifest(
    output_dir: Path | str, manifest: PrmCheckpointManifest
) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    target = p / MANIFEST_FILENAME
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


def read_manifest(path: Path | str) -> PrmCheckpointManifest:
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return PrmCheckpointManifest.from_dict(payload)


def write_training_config(
    output_dir: Path | str, training_config: Mapping[str, Any]
) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    target = p / TRAINING_CONFIG_FILENAME
    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(training_config), f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


R2Uploader = Callable[[Path, str], str]
"""Same signature as Phase 29
:type:`vlabs_reward_train.checkpointing.R2Uploader`."""


def fake_r2_uploader(target_dir: Path | str = "/tmp/fake-r2") -> R2Uploader:
    """In-memory R2 uploader for tests + dry-run CLI path."""
    base = Path(target_dir)

    def upload(local_path: Path, r2_key: str) -> str:
        if r2_key.startswith("/"):
            r2_key = r2_key.lstrip("/")
        dest = base / r2_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local_path.read_bytes())
        return f"r2://vlabs-models/{r2_key}"

    return upload


def upload_checkpoint(
    output_dir: Path | str,
    manifest: PrmCheckpointManifest,
    *,
    uploader: R2Uploader | None = None,
) -> dict[str, str]:
    """Upload every file listed in the manifest + the manifest itself.

    Returns a ``{filename: r2_uri}`` map. ``uploader=None`` defaults
    to :func:`fake_r2_uploader` writing to a temp tree."""
    if uploader is None:
        uploader = fake_r2_uploader()
    base = Path(output_dir)
    if not base.exists():
        raise FileNotFoundError(f"checkpoint dir {base!r} does not exist")

    uploaded: dict[str, str] = {}
    for filename in (*manifest.checkpoint_files, MANIFEST_FILENAME):
        local = base / filename
        if not local.exists():
            logger.warning("skipping missing file %s", local)
            continue
        r2_key = f"{manifest.model_id}/{manifest.version}/checkpoint/{filename}"
        uploaded[filename] = uploader(local, r2_key)
    return uploaded


def list_local_checkpoints(parent_dir: Path | str) -> list[PrmCheckpointManifest]:
    base = Path(parent_dir)
    if not base.exists():
        return []
    manifests: list[PrmCheckpointManifest] = []
    for path in sorted(base.rglob(MANIFEST_FILENAME)):
        try:
            manifests.append(read_manifest(path))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("failed to read PRM manifest %s: %s", path, exc)
            continue
    return manifests


def manifest_table(
    manifests: Iterable[PrmCheckpointManifest],
) -> list[dict[str, Any]]:
    """Pretty-printable table view — surfaces the moat-relevant fields
    (model_id, version, base_rm_id, calibration quantile, fingerprint)
    and skips the heavier nested configs."""
    rows: list[dict[str, Any]] = []
    for m in manifests:
        rows.append(
            {
                "model_id": m.model_id,
                "version": m.version,
                "base_model": m.base_model,
                "base_rm_id": m.base_rm_id,
                "step_granularity": m.step_granularity,
                "aggregate_conformal_quantile": m.aggregate_conformal_quantile,
                "step_buckets": (
                    sorted(m.step_conformal_quantiles)
                    if m.step_conformal_quantiles
                    else []
                ),
                "fingerprint": m.fingerprint[:16],
                "n_files": len(m.checkpoint_files),
            }
        )
    return rows


def is_real_r2_configured() -> bool:
    return bool(os.environ.get("R2_ACCESS_KEY_ID")) and bool(
        os.environ.get("R2_SECRET_ACCESS_KEY")
    )


__all__ = [
    "LORA_WEIGHTS_FILENAME",
    "MANIFEST_FILENAME",
    "R2Uploader",
    "SCHEMA_VERSION",
    "TRAINING_CONFIG_FILENAME",
    "PrmCheckpointManifest",
    "fake_r2_uploader",
    "is_real_r2_configured",
    "list_local_checkpoints",
    "manifest_table",
    "model_id_for",
    "read_manifest",
    "upload_checkpoint",
    "write_manifest",
    "write_training_config",
]
