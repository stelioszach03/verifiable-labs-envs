"""Public-facing ID helpers — UUID encoded with a stable prefix.

The DB stores UUIDs natively. The public API exposes them as
``cal_<32-char hex>`` (and similar) so callers can grep their logs and
distinguish ID kinds at a glance.
"""
from __future__ import annotations

import uuid

from vlabs_api.errors import (
    AuditCallNotFound,
    CalibrationNotFound,
    DatasetJobNotFound,
    MonitorNotFound,
    MonitorRunNotFound,
)

CALIBRATION_PREFIX = "cal_"
AUDIT_PREFIX = "aud_"
DATASET_PREFIX = "ds_"
MONITOR_PREFIX = "mon_"
MONITOR_RUN_PREFIX = "mr_"


def encode_calibration_id(uid: uuid.UUID) -> str:
    return f"{CALIBRATION_PREFIX}{uid.hex}"


def parse_calibration_id(s: str) -> uuid.UUID:
    """Accept either ``cal_<hex>`` (preferred) or a bare UUID string.

    Raises :class:`CalibrationNotFound` on any parse failure — same
    surface as a missing row, since both are user-error and we
    don't leak which is which.
    """
    raw = s[len(CALIBRATION_PREFIX):] if s.startswith(CALIBRATION_PREFIX) else s
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise CalibrationNotFound(detail=f"invalid calibration_id: {s!r}") from exc


def encode_audit_id(uid: uuid.UUID) -> str:
    """Phase 22.D — public ID for ``audit_calls`` rows."""
    return f"{AUDIT_PREFIX}{uid.hex}"


def parse_audit_id(s: str) -> uuid.UUID:
    """Phase 22.D — inverse of :func:`encode_audit_id`.

    Accepts either ``aud_<hex>`` (preferred) or a bare UUID string.
    Raises :class:`AuditCallNotFound` on any parse failure (same
    information-hiding posture as :func:`parse_calibration_id`).
    """
    raw = s[len(AUDIT_PREFIX):] if s.startswith(AUDIT_PREFIX) else s
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise AuditCallNotFound(detail=f"invalid audit_id: {s!r}") from exc


def encode_dataset_id(uid: uuid.UUID) -> str:
    """Phase 23.B — public ID for ``dataset_jobs`` rows."""
    return f"{DATASET_PREFIX}{uid.hex}"


def parse_dataset_id(s: str) -> uuid.UUID:
    """Phase 23.B — inverse of :func:`encode_dataset_id`.

    Accepts either ``ds_<hex>`` (preferred) or a bare UUID string.
    Raises :class:`DatasetJobNotFound` on parse failure (matches the
    information-hiding posture of the other parse_*_id helpers).
    """
    raw = s[len(DATASET_PREFIX):] if s.startswith(DATASET_PREFIX) else s
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise DatasetJobNotFound(detail=f"invalid dataset_id: {s!r}") from exc


def encode_monitor_id(uid: uuid.UUID) -> str:
    """Phase 28.B — public ID for ``monitors`` rows."""
    return f"{MONITOR_PREFIX}{uid.hex}"


def parse_monitor_id(s: str) -> uuid.UUID:
    """Phase 28.B — inverse of :func:`encode_monitor_id`."""
    raw = s[len(MONITOR_PREFIX):] if s.startswith(MONITOR_PREFIX) else s
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise MonitorNotFound(detail=f"invalid monitor_id: {s!r}") from exc


def encode_monitor_run_id(uid: uuid.UUID) -> str:
    """Phase 28.B — public ID for ``monitor_runs`` rows."""
    return f"{MONITOR_RUN_PREFIX}{uid.hex}"


def parse_monitor_run_id(s: str) -> uuid.UUID:
    """Phase 28.B — inverse of :func:`encode_monitor_run_id`."""
    raw = (
        s[len(MONITOR_RUN_PREFIX):]
        if s.startswith(MONITOR_RUN_PREFIX)
        else s
    )
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise MonitorRunNotFound(detail=f"invalid monitor_run_id: {s!r}") from exc


__all__ = [
    "CALIBRATION_PREFIX",
    "AUDIT_PREFIX",
    "DATASET_PREFIX",
    "MONITOR_PREFIX",
    "MONITOR_RUN_PREFIX",
    "encode_calibration_id",
    "parse_calibration_id",
    "encode_audit_id",
    "parse_audit_id",
    "encode_dataset_id",
    "parse_dataset_id",
    "encode_monitor_id",
    "parse_monitor_id",
    "encode_monitor_run_id",
    "parse_monitor_run_id",
]
