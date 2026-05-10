"""Public-facing ID helpers — UUID encoded with a stable prefix.

The DB stores UUIDs natively. The public API exposes them as
``cal_<32-char hex>`` (and similar) so callers can grep their logs and
distinguish ID kinds at a glance.
"""
from __future__ import annotations

import uuid

from vlabs_api.errors import (
    AttestationNotFound,
    AuditCallNotFound,
    CalibrationNotFound,
    DatasetJobNotFound,
    MonitorNotFound,
    MonitorRunNotFound,
    ProcessRewardModelNotFound,
    RewardModelNotFound,
)

CALIBRATION_PREFIX = "cal_"
AUDIT_PREFIX = "aud_"
DATASET_PREFIX = "ds_"
MONITOR_PREFIX = "mon_"
MONITOR_RUN_PREFIX = "mr_"
REWARD_MODEL_RUN_PREFIX = "rmr_"
PROCESS_REWARD_RUN_PREFIX = "prr_"
ATTESTATION_PREFIX = "att_"
ATTESTATION_ARTIFACT_PREFIX = "attart_"
ATTESTATION_AUDIT_PREFIX = "attaud_"
ATTESTATION_RENEWAL_PREFIX = "attren_"
ATTESTATION_PUBLIC_ID_PREFIX = "vl-"
ATTESTATION_PUBLIC_ID_LEN = 8
"""Phase 31 D5-C / D11 — public_id is the short URL-safe identifier
surfaced on verify.verifiable-labs.com. Shape: ``vl-XXXXXXXX`` with
8 base32 chars (Crockford alphabet) ⇒ 40 bits of entropy ⇒
~1.1 trillion possible values, vastly more than the eventual customer
population. Generated deterministically from the internal UUID for
reproducibility."""


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


_CROCKFORD_ALPHABET: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""Crockford base32 alphabet — omits I, L, O, U so public_id codes
are unambiguous when read aloud or transcribed."""


def encode_attestation_id(uid: uuid.UUID) -> str:
    """Phase 31.B — owner-facing UUID-shaped attestation id.

    Shape: ``att_<32-char-hex>``. Used in
    ``GET /v1/attestations/{id}`` (X-Vlabs-Key auth required).
    """
    return f"{ATTESTATION_PREFIX}{uid.hex}"


def parse_attestation_id(s: str) -> uuid.UUID:
    """Phase 31.B — inverse of :func:`encode_attestation_id`."""
    raw = (
        s[len(ATTESTATION_PREFIX):]
        if s.startswith(ATTESTATION_PREFIX)
        else s
    )
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise AttestationNotFound(
            detail=f"invalid attestation id: {s!r}"
        ) from exc


def encode_attestation_artifact_id(uid: uuid.UUID) -> str:
    """Phase 31.B — public id for ``attestation_artifacts`` rows.

    Shape: ``attart_<32-char-hex>``. Returned in artifact-upload
    responses so the customer can later reference / re-fetch the
    artifact.
    """
    return f"{ATTESTATION_ARTIFACT_PREFIX}{uid.hex}"


def encode_attestation_audit_id(uid: uuid.UUID) -> str:
    """Phase 31.B — public id for ``attestation_audits`` rows.

    Shape: ``attaud_<32-char-hex>``. Surfaced in the audit-trail
    sub-response so customers can correlate decisions with
    individual auditors (D12 / R6 multi-party-approval transparency).
    """
    return f"{ATTESTATION_AUDIT_PREFIX}{uid.hex}"


def encode_attestation_renewal_id(uid: uuid.UUID) -> str:
    """Phase 31.B — public id for ``attestation_renewals`` rows.

    Shape: ``attren_<32-char-hex>``. Surfaced in renewal-initiation
    responses so the customer can poll progress.
    """
    return f"{ATTESTATION_RENEWAL_PREFIX}{uid.hex}"


def encode_attestation_public_id(uid: uuid.UUID) -> str:
    """Phase 31.D / D5-C / D11 — short URL-safe public verification ID.

    Shape: ``vl-XXXXXXXX`` with 8 Crockford-base32 chars derived
    deterministically from the upper 40 bits of the input UUID.
    Same UUID always maps to the same public_id (collision-resistant
    enough at our customer scale; the `attestations.public_id` UNIQUE
    index catches any collision at insert time and the service layer
    retries with a different seed).

    Crockford base32 omits ``I L O U`` so ``vl-`` codes are
    unambiguous when transcribed by hand or read aloud.
    """
    raw_bytes = uid.bytes[:5]  # 40 bits
    value = int.from_bytes(raw_bytes, "big")
    chars: list[str] = []
    for _ in range(ATTESTATION_PUBLIC_ID_LEN):
        chars.append(_CROCKFORD_ALPHABET[value & 0x1F])
        value >>= 5
    chars.reverse()
    return f"{ATTESTATION_PUBLIC_ID_PREFIX}{''.join(chars)}"


def parse_attestation_public_id(s: str) -> str:
    """Phase 31.D — validate the public_id shape + return the bare
    8-char Crockford code (without the ``vl-`` prefix). Used by the
    public verification endpoint for lookup against
    ``attestations.public_id`` UNIQUE index.

    Accepts either ``vl-XXXXXXXX`` or bare ``XXXXXXXX``; rejects
    anything outside the Crockford alphabet.
    """
    raw = (
        s[len(ATTESTATION_PUBLIC_ID_PREFIX):]
        if s.startswith(ATTESTATION_PUBLIC_ID_PREFIX)
        else s
    )
    if len(raw) != ATTESTATION_PUBLIC_ID_LEN:
        raise AttestationNotFound(detail=f"invalid public_id: {s!r}")
    raw_upper = raw.upper()
    if not all(c in _CROCKFORD_ALPHABET for c in raw_upper):
        raise AttestationNotFound(detail=f"invalid public_id: {s!r}")
    return raw_upper


def encode_process_reward_run_id(uid: uuid.UUID) -> str:
    """Phase 30.E — public ID for ``process_reward_model_runs`` rows.

    Shape: ``prr_<32-char-hex>``. Surfaced as ``audit_id`` in
    ``POST /v1/process-reward-models/{id}/score`` responses so
    customers can later reference the run from
    ``GET /v1/process-reward-models/{id}/evals``.
    """
    return f"{PROCESS_REWARD_RUN_PREFIX}{uid.hex}"


def parse_process_reward_run_id(s: str) -> uuid.UUID:
    """Phase 30.E — inverse of :func:`encode_process_reward_run_id`."""
    raw = (
        s[len(PROCESS_REWARD_RUN_PREFIX):]
        if s.startswith(PROCESS_REWARD_RUN_PREFIX)
        else s
    )
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise ProcessRewardModelNotFound(
            detail=f"invalid run_id: {s!r}"
        ) from exc


def encode_reward_model_run_id(uid: uuid.UUID) -> str:
    """Phase 29.E — public ID for ``reward_model_runs`` rows.

    Shape: ``rmr_<32-char-hex>``. Surfaced as ``audit_id`` in
    ``POST /v1/reward-models/{id}/score`` responses so customers can
    later reference the run from ``GET /v1/reward-models/{id}/evals``.
    """
    return f"{REWARD_MODEL_RUN_PREFIX}{uid.hex}"


def parse_reward_model_run_id(s: str) -> uuid.UUID:
    """Phase 29.E — inverse of :func:`encode_reward_model_run_id`."""
    raw = (
        s[len(REWARD_MODEL_RUN_PREFIX):]
        if s.startswith(REWARD_MODEL_RUN_PREFIX)
        else s
    )
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise RewardModelNotFound(detail=f"invalid run_id: {s!r}") from exc


__all__ = [
    "CALIBRATION_PREFIX",
    "AUDIT_PREFIX",
    "DATASET_PREFIX",
    "MONITOR_PREFIX",
    "MONITOR_RUN_PREFIX",
    "REWARD_MODEL_RUN_PREFIX",
    "PROCESS_REWARD_RUN_PREFIX",
    "ATTESTATION_PREFIX",
    "ATTESTATION_ARTIFACT_PREFIX",
    "ATTESTATION_AUDIT_PREFIX",
    "ATTESTATION_RENEWAL_PREFIX",
    "ATTESTATION_PUBLIC_ID_PREFIX",
    "ATTESTATION_PUBLIC_ID_LEN",
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
    "encode_reward_model_run_id",
    "parse_reward_model_run_id",
    "encode_process_reward_run_id",
    "parse_process_reward_run_id",
    "encode_attestation_id",
    "parse_attestation_id",
    "encode_attestation_artifact_id",
    "encode_attestation_audit_id",
    "encode_attestation_renewal_id",
    "encode_attestation_public_id",
    "parse_attestation_public_id",
]
