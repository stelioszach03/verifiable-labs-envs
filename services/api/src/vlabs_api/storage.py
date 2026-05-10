"""Object-storage backend for vlabs-data datasets (Phase 23.C).

PHASE_23_PLAN.md §8 ruling: Cloudflare R2 (S3-compatible) for prod.
LOCAL_FAKE_R2 mode writes to ``/tmp/r2-fake/<bucket>/<key>`` for tests
+ local dev — same interface, no network calls.

Object key convention: ``{user_id}/{dataset_id}/{format}.{ext}``.

Migration path away from R2 (R5 mitigation): change endpoint URL +
access keys; object keys stay identical. Zero application code change.
"""
from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

from vlabs_api.config import get_settings
from vlabs_api.errors import R2StorageError

log = structlog.get_logger(__name__)

# LOCAL_FAKE_R2 storage root.
_FAKE_R2_ROOT = Path("/tmp/r2-fake")

# Default signed-URL TTL.
DEFAULT_SIGNED_URL_TTL = timedelta(hours=1)


def _is_fake_mode() -> bool:
    return get_settings().vlabs_local_fake_r2


def _bucket_name() -> str:
    return get_settings().vlabs_r2_bucket_name


def _resolve_endpoint() -> str:
    """Compute the R2 endpoint URL — explicit override or auto-derived."""
    s = get_settings()
    if s.vlabs_r2_endpoint_url:
        return s.vlabs_r2_endpoint_url
    if not s.vlabs_r2_account_id:
        raise R2StorageError(
            detail="R2 endpoint not configured: VLABS_R2_ENDPOINT_URL "
            "and VLABS_R2_ACCOUNT_ID are both empty"
        )
    return f"https://{s.vlabs_r2_account_id}.r2.cloudflarestorage.com"


# Map a logical ``output_format`` name to a concrete on-disk
# extension. The previous implementation treated anything other than
# ``"parquet"`` as ``"jsonl"``, which produced filenames like
# ``pdf.jsonl`` for monitor PDF artefacts (Phase 28 validation report,
# reports/validation/SUMMARY.md §"Storage _build_key PDF extension").
#
# The fix: extend the format→ext map to cover every output format we
# generate today, and fall back to the format string itself when an
# unknown format is requested (so future formats produce sensible
# names without a code change). The presigned-URL content-type header
# stays correct in either case — that's set per-upload — but the on-
# disk filename now matches the format.
_EXTENSION_BY_FORMAT: dict[str, str] = {
    "parquet": "parquet",
    "jsonl": "jsonl",
    "pdf": "pdf",
    "csv": "csv",
    "json": "json",
}


def _ext_for_format(output_format: str) -> str:
    """Return the on-disk extension for ``output_format``.

    Unknown formats fall back to the format string itself (lower-cased,
    stripped of any leading ``.``). This keeps ``_build_key`` resilient
    to new formats without hard-coding a closed set.
    """
    if output_format in _EXTENSION_BY_FORMAT:
        return _EXTENSION_BY_FORMAT[output_format]
    cleaned = output_format.lstrip(".").lower().strip()
    return cleaned or "bin"


def _build_key(user_id: str, dataset_id: str, output_format: str) -> str:
    """Object key for a dataset payload."""
    return f"{user_id}/{dataset_id}/{output_format}.{_ext_for_format(output_format)}"


def _fake_path(key: str) -> Path:
    return _FAKE_R2_ROOT / _bucket_name() / key


def _real_client() -> Any:
    """Build a boto3 S3 client pointing at R2.

    Lazy import — boto3 is heavy and only needed when LOCAL_FAKE_R2 is
    off. Tests never load this path.
    """
    import boto3

    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=_resolve_endpoint(),
        aws_access_key_id=s.vlabs_r2_access_key_id,
        aws_secret_access_key=s.vlabs_r2_secret_access_key,
        region_name="auto",
    )


def upload_dataset(
    user_id: str,
    dataset_id: str,
    output_format: str,
    payload: bytes,
) -> tuple[str, str, int]:
    """Upload a complete dataset payload.

    Returns ``(storage_key, sha256, size_bytes)``. Computes SHA-256 of
    the bytes BEFORE upload so the integrity hash is exact even if R2
    rewrites the body during transit (D10-A guarantee).
    """
    key = _build_key(user_id, dataset_id, output_format)
    sha256 = hashlib.sha256(payload).hexdigest()
    size_bytes = len(payload)

    if _is_fake_mode():
        path = _fake_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        log.info(
            "storage.fake_upload",
            key=key,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        return key, sha256, size_bytes

    try:
        client = _real_client()
        client.put_object(
            Bucket=_bucket_name(),
            Key=key,
            Body=payload,
            ContentType=(
                "application/x-parquet"
                if output_format == "parquet"
                else "application/x-jsonlines"
            ),
            Metadata={"sha256": sha256},
        )
    except Exception as exc:  # noqa: BLE001
        raise R2StorageError(detail=f"R2 put_object failed: {exc}") from exc

    log.info(
        "storage.real_upload",
        key=key,
        size_bytes=size_bytes,
        sha256=sha256,
    )
    return key, sha256, size_bytes


def upload_chunk(
    user_id: str,
    dataset_id: str,
    output_format: str,
    chunk_idx: int,
    payload: bytes,
) -> str:
    """Upload one checkpoint chunk during in-progress generation.

    Returns the chunk's object key. Chunks are concatenated into the
    final dataset by :func:`finalize_chunks` when generation completes.
    """
    base_key = _build_key(user_id, dataset_id, output_format)
    key = f"{base_key}.chunk-{chunk_idx:06d}"
    if _is_fake_mode():
        path = _fake_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return key

    try:
        client = _real_client()
        client.put_object(
            Bucket=_bucket_name(),
            Key=key,
            Body=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise R2StorageError(detail=f"R2 chunk upload failed: {exc}") from exc
    return key


def list_chunks(
    user_id: str,
    dataset_id: str,
    output_format: str,
) -> list[str]:
    """List all chunk keys for a job, sorted ascending by chunk_idx."""
    base_key = _build_key(user_id, dataset_id, output_format)
    prefix = f"{base_key}.chunk-"

    if _is_fake_mode():
        bucket_root = _FAKE_R2_ROOT / _bucket_name()
        if not bucket_root.exists():
            return []
        return sorted(
            str(p.relative_to(bucket_root))
            for p in bucket_root.rglob("*")
            if p.is_file() and str(p.relative_to(bucket_root)).startswith(prefix)
        )

    try:
        client = _real_client()
        out: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_bucket_name(), Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                out.append(obj["Key"])
        return sorted(out)
    except Exception as exc:  # noqa: BLE001
        raise R2StorageError(detail=f"R2 list_objects failed: {exc}") from exc


def delete_chunks(
    user_id: str,
    dataset_id: str,
    output_format: str,
) -> int:
    """Delete all chunks after the final upload. Returns count deleted."""
    keys = list_chunks(user_id, dataset_id, output_format)
    if not keys:
        return 0
    if _is_fake_mode():
        for key in keys:
            (_FAKE_R2_ROOT / _bucket_name() / key).unlink(missing_ok=True)
        return len(keys)
    try:
        client = _real_client()
        for key in keys:
            client.delete_object(Bucket=_bucket_name(), Key=key)
    except Exception as exc:  # noqa: BLE001
        raise R2StorageError(detail=f"R2 chunk cleanup failed: {exc}") from exc
    return len(keys)


def generate_signed_url(
    storage_key: str,
    ttl: timedelta = DEFAULT_SIGNED_URL_TTL,
) -> tuple[str, datetime]:
    """Generate a presigned download URL.

    Returns ``(url, expires_at)``. In LOCAL_FAKE_R2 mode the "URL" is
    a ``file://`` reference to the on-disk fake path — handy for tests
    that want to sanity-check the contents.
    """
    expires_at = datetime.now(UTC) + ttl
    if _is_fake_mode():
        path = _fake_path(storage_key)
        return f"file://{path}", expires_at
    try:
        client = _real_client()
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": _bucket_name(), "Key": storage_key},
            ExpiresIn=int(ttl.total_seconds()),
        )
    except Exception as exc:  # noqa: BLE001
        raise R2StorageError(detail=f"R2 presigned URL failed: {exc}") from exc
    return url, expires_at


def reset_fake_storage_for_tests() -> None:
    """Wipe ``/tmp/r2-fake/<bucket>/`` — only safe to call from tests."""
    bucket_root = _FAKE_R2_ROOT / _bucket_name()
    if bucket_root.exists():
        shutil.rmtree(bucket_root)


__all__ = [
    "DEFAULT_SIGNED_URL_TTL",
    "upload_dataset",
    "upload_chunk",
    "list_chunks",
    "delete_chunks",
    "generate_signed_url",
    "reset_fake_storage_for_tests",
]
