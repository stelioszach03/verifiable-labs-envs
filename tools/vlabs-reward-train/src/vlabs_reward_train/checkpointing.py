"""Checkpoint persistence for the trained student (29.C scaffold).

Per :doc:`PHASE_29_PLAN.md` §5 D12-B: each checkpoint is identified by
a semver `model_id` and persisted both locally (under ``output_dir``)
and to R2 (``r2://vlabs-models/{model_id}/{version}/checkpoint/``).

In 29.C the local path is fully exercised + tested; the R2 upload is
**stubbed** behind a callable so 29.G can swap in the real
``vlabs_api.storage`` client without touching this module's contract.

Manifest shape (canonical JSON, sort_keys=True so it's diffable):

```json
{
  "model_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
  "version": "0.1.0",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "lora_config": {...},
  "training_config": {...},
  "metrics": {...},
  "checkpoint_files": [...],
  "conformal_quantile": 0.087,
  "schema_version": "v0.1.0",
  "created_at_unix": 1234567890
}
```
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
class CheckpointManifest:
    """Canonical record of one trained checkpoint.

    Persisted under ``{output_dir}/manifest.json``; uploaded to R2 next
    to the LoRA weights.
    """

    model_id: str
    version: str
    base_model: str
    lora_config: dict[str, Any]
    training_config: dict[str, Any]
    metrics: dict[str, Any]
    checkpoint_files: tuple[str, ...]
    conformal_quantile: float | None = None
    created_at_unix: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["checkpoint_files"] = list(self.checkpoint_files)
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointManifest:
        return cls(
            model_id=str(payload["model_id"]),
            version=str(payload["version"]),
            base_model=str(payload["base_model"]),
            lora_config=dict(payload.get("lora_config", {})),
            training_config=dict(payload.get("training_config", {})),
            metrics=dict(payload.get("metrics", {})),
            checkpoint_files=tuple(payload.get("checkpoint_files", ())),
            conformal_quantile=payload.get("conformal_quantile"),
            created_at_unix=float(payload.get("created_at_unix", time.time())),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint over the manifest (excluding the
        timestamp). Used as the audit-trail row id for trained
        checkpoints in 29.G."""
        payload = {k: v for k, v in self.to_dict().items() if k != "created_at_unix"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def model_id_for(family: str, version: str) -> str:
    """Format the locked D12-B model id shape:
    ``vlabs-reward-{family}-v{semver}``.

    Example: ``vlabs-reward-distilled-qwen-1-5b-v0.1.0``.
    """
    if not family:
        raise ValueError("family must be non-empty")
    if not version:
        raise ValueError("version must be non-empty")
    return f"vlabs-reward-{family}-v{version}"


def write_manifest(
    output_dir: Path | str, manifest: CheckpointManifest
) -> Path:
    """Write the manifest JSON to ``{output_dir}/manifest.json``."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    target = p / MANIFEST_FILENAME
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


def read_manifest(path: Path | str) -> CheckpointManifest:
    """Inverse of :func:`write_manifest`."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return CheckpointManifest.from_dict(payload)


def write_training_config(
    output_dir: Path | str, training_config: Mapping[str, Any]
) -> Path:
    """Persist the training-time hyperparameters next to the
    checkpoint. Mirrors the W&B run-config payload but is the source
    of truth on disk."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    target = p / TRAINING_CONFIG_FILENAME
    with target.open("w", encoding="utf-8") as f:
        json.dump(dict(training_config), f, sort_keys=True, ensure_ascii=False, indent=2)
    return target


R2Uploader = Callable[[Path, str], str]
"""Callable signature: ``(local_path, r2_key) -> r2_uri``.

29.G replaces the stub with a real `vlabs_api.storage` adapter. Tests
inject :func:`fake_r2_uploader` so the upload path round-trips
bit-identically without network."""


def fake_r2_uploader(target_dir: Path | str = "/tmp/fake-r2") -> R2Uploader:
    """Return an in-memory R2 uploader that copies files into a local
    tree under ``target_dir``. Used by tests + the dry-run CLI path
    to exercise the upload contract end-to-end."""
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
    manifest: CheckpointManifest,
    *,
    uploader: R2Uploader | None = None,
) -> dict[str, str]:
    """Upload every file listed in the manifest + the manifest itself.

    Returns a ``{filename: r2_uri}`` map. ``uploader=None`` defaults to
    :func:`fake_r2_uploader` writing to a temp tree — keeps the dry-run
    CLI path functional without env credentials.
    """
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


def list_local_checkpoints(parent_dir: Path | str) -> list[CheckpointManifest]:
    """Discover all checkpoints under ``parent_dir`` by recursively
    looking for ``manifest.json`` files. Used by the CLI's
    ``checkpoints list`` subcommand and by 29.D's eval harness when
    sweeping multiple runs.
    """
    base = Path(parent_dir)
    if not base.exists():
        return []
    manifests: list[CheckpointManifest] = []
    for path in sorted(base.rglob(MANIFEST_FILENAME)):
        try:
            manifests.append(read_manifest(path))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("failed to read manifest %s: %s", path, exc)
            continue
    return manifests


def manifest_table(manifests: Iterable[CheckpointManifest]) -> list[dict[str, Any]]:
    """Pretty-printable table view of a manifest collection. Each row
    surfaces the moat-relevant fields (model_id, version, status,
    quantile, fingerprint) and skips the heavier nested configs."""
    rows: list[dict[str, Any]] = []
    for m in manifests:
        rows.append(
            {
                "model_id": m.model_id,
                "version": m.version,
                "base_model": m.base_model,
                "conformal_quantile": m.conformal_quantile,
                "fingerprint": m.fingerprint[:16],
                "n_files": len(m.checkpoint_files),
                "metrics_keys": sorted(m.metrics),
            }
        )
    return rows


def is_real_r2_configured() -> bool:
    """Predicate: are R2 credentials available in the environment?

    Mirrors the gating contract — if they aren't, callers should
    default to :func:`fake_r2_uploader`.
    """
    return bool(os.environ.get("R2_ACCESS_KEY_ID")) and bool(
        os.environ.get("R2_SECRET_ACCESS_KEY")
    )


__all__ = [
    "LORA_WEIGHTS_FILENAME",
    "MANIFEST_FILENAME",
    "R2Uploader",
    "SCHEMA_VERSION",
    "TRAINING_CONFIG_FILENAME",
    "CheckpointManifest",
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
