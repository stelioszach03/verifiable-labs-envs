"""V-Certified CRL (Certificate Revocation List) generation (Phase 31.D).

The public CRL is served as a signed PEM at ``/v1/attestations/crl.pem``.
Verifiers fetch it, validate the signature against the V-Certified CA
public key, then check whether their target attestation's cert serial
appears in the revoked-certs list.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from vlabs_api.pki.cert_issuer import _serial_from_str
from vlabs_api.pki.fake_hsm import FakeHSMBackend, get_default_backend

CRL_NEXT_UPDATE_DAYS = 1


def build_crl_pem(
    *,
    revoked: Sequence[tuple[str, datetime]],
    backend: FakeHSMBackend | None = None,
) -> str:
    """Build a fresh signed CRL covering the supplied revoked serials.

    Parameters
    ----------
    revoked
        Sequence of ``(cert_serial, revocation_date)`` pairs. The
        cert_serial is opaque to verifiers but must be the same value
        embedded in the leaf certs' OU attribute.
    """
    backend = backend or get_default_backend()
    now = datetime.now(UTC)

    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(backend.ca_certificate.subject)
        .last_update(now)
        .next_update(now + timedelta(days=CRL_NEXT_UPDATE_DAYS))
    )
    for cert_serial, revoked_at in revoked:
        # Normalise to UTC-aware to satisfy cryptography's tz check.
        if revoked_at.tzinfo is None:
            revoked_at = revoked_at.replace(tzinfo=UTC)
        revoked_cert = (
            x509.RevokedCertificateBuilder()
            .serial_number(_serial_from_str(cert_serial))
            .revocation_date(revoked_at)
            .build()
        )
        builder = builder.add_revoked_certificate(revoked_cert)

    crl = builder.sign(backend.ca_private_key, hashes.SHA256())
    return crl.public_bytes(serialization.Encoding.PEM).decode("ascii")


def parse_crl_serials(crl_pem: str) -> list[int]:
    """Return the integer serial numbers in a CRL.

    Useful for tests asserting which certs are in the revoked list.
    """
    crl = x509.load_pem_x509_crl(crl_pem.encode("ascii"))
    return [int(rc.serial_number) for rc in crl]


def verify_crl_signature(
    crl_pem: str, *, backend: FakeHSMBackend | None = None
) -> bool:
    """Verify a CRL was signed by the V-Certified CA."""
    backend = backend or get_default_backend()
    try:
        crl = x509.load_pem_x509_crl(crl_pem.encode("ascii"))
    except ValueError:
        return False
    return crl.is_signature_valid(backend.ca_private_key.public_key())


__all__ = [
    "CRL_NEXT_UPDATE_DAYS",
    "build_crl_pem",
    "parse_crl_serials",
    "verify_crl_signature",
]
