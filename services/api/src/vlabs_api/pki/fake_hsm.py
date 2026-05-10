"""Fake HSM backend for the V-Certified CA (Phase 31.D, test-only).

Generates a single in-memory RSA-2048 CA keypair + self-signed CA
certificate on first use, then signs every leaf cert with it. Key
material lives only in process memory and is reset between tests via
:func:`reset_default_backend_for_tests`.

Production must instead use :mod:`vlabs_api.pki.kms_hsm` (deferred), so
the unwrapped CA private key never enters the API process.

Activation is gated on the VLABS_LOCAL_FAKE_PKI env var — production
deploys MUST leave this unset; the api boot will refuse to load the
fake backend if VLABS_ENVIRONMENT is "prod".
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CA_COMMON_NAME = "Verifiable Labs V-Certified CA"
CA_ORGANIZATION = "Verifiable Labs, Inc."
CA_COUNTRY = "US"
CA_VALIDITY_DAYS = 3650  # 10 years (test-only).
CA_KEY_BITS = 2048


class FakeHSMBackend:
    """In-memory CA keypair + self-signed certificate.

    Constructed once per process; safe to share across requests
    because the cryptography APIs are thread-safe for sign-only ops.
    """

    def __init__(self) -> None:
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=CA_KEY_BITS
        )
        self._certificate = self._build_self_signed()

    def _build_self_signed(self) -> x509.Certificate:
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, CA_COMMON_NAME),
                x509.NameAttribute(
                    NameOID.ORGANIZATION_NAME, CA_ORGANIZATION
                ),
                x509.NameAttribute(NameOID.COUNTRY_NAME, CA_COUNTRY),
            ]
        )
        now = datetime.now(UTC)
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self._private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=CA_VALIDITY_DAYS))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        )
        return builder.sign(self._private_key, hashes.SHA256())

    @property
    def ca_certificate(self) -> x509.Certificate:
        return self._certificate

    @property
    def ca_certificate_pem(self) -> str:
        return self._certificate.public_bytes(
            serialization.Encoding.PEM
        ).decode("ascii")

    @property
    def ca_private_key(self) -> rsa.RSAPrivateKey:
        return self._private_key


_default: FakeHSMBackend | None = None


def get_default_backend() -> FakeHSMBackend:
    """Return the process-wide singleton fake-HSM backend.

    Refuses to initialise in production environments (refuses if
    VLABS_ENVIRONMENT=prod and VLABS_LOCAL_FAKE_PKI is not set).
    """
    global _default
    if _default is None:
        env = os.environ.get("VLABS_ENVIRONMENT", "dev").lower()
        fake_ok = (
            os.environ.get("VLABS_LOCAL_FAKE_PKI", "").lower() == "true"
        )
        if env == "prod" and not fake_ok:
            raise RuntimeError(
                "fake_hsm refused: VLABS_ENVIRONMENT=prod requires the "
                "kms_hsm backend (Phase 31 production-hardening track) "
                "or VLABS_LOCAL_FAKE_PKI=true override"
            )
        _default = FakeHSMBackend()
    return _default


def reset_default_backend_for_tests() -> None:
    """Clear the cached singleton so the next get_default_backend()
    call re-generates a fresh CA. Used by the conftest truncate hook
    to keep tests deterministic."""
    global _default
    _default = None


__all__ = [
    "CA_COMMON_NAME",
    "CA_ORGANIZATION",
    "CA_VALIDITY_DAYS",
    "FakeHSMBackend",
    "get_default_backend",
    "reset_default_backend_for_tests",
]
