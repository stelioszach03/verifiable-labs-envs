"""Public-facing ID helpers — UUID encoded with a stable prefix.

The DB stores UUIDs natively. The public API exposes them as
``cal_<32-char hex>`` (and similar) so callers can grep their logs and
distinguish ID kinds at a glance.
"""
from __future__ import annotations

import uuid

from vlabs_api.errors import CalibrationNotFound

CALIBRATION_PREFIX = "cal_"


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


__all__ = ["CALIBRATION_PREFIX", "encode_calibration_id", "parse_calibration_id"]
