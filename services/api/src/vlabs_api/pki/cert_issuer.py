"""Issue + parse + verify V-Certified leaf certificates (Phase 31.D).

Each approved attestation gets one leaf certificate signed by the
V-Certified CA. The cert's CN encodes the public_id (vl-XXXXXXXX) and
the OU encodes the cert serial. Lifetime mirrors the attestation
status — annual cycles get 365 days, continuous cycles get 395 days,
matching the lifetime windows in attestation_service.

Verification is done by:
1. parsing the leaf cert's DER bytes;
2. checking the issuer matches the CA cert subject;
3. validating the signature against the CA public key.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from vlabs_api.pki.fake_hsm import (
    CA_COMMON_NAME,
    FakeHSMBackend,
    get_default_backend,
)

LEAF_KEY_BITS = 2048


def issue_leaf_certificate(
    *,
    public_id: str,
    organization: str,
    cert_serial: str,
    lifetime: timedelta,
    backend: FakeHSMBackend | None = None,
) -> tuple[str, str]:
    """Issue a fresh leaf certificate for an approved attestation.

    Returns ``(certificate_pem, private_key_pem)``. The private key
    is fresh per-leaf and only returned to the caller (the V-Certified
    programme retains the public cert; the customer keeps the private
    key for any TLS use cases).
    """
    backend = backend or get_default_backend()
    leaf_key = rsa.generate_private_key(
        public_exponent=65537, key_size=LEAF_KEY_BITS
    )
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, public_id),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, cert_serial),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        ]
    )
    issuer = backend.ca_certificate.subject

    now = datetime.now(UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(leaf_key.public_key())
        .serial_number(_serial_from_str(cert_serial))
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + lifetime)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    )

    leaf_cert = builder.sign(backend.ca_private_key, hashes.SHA256())
    cert_pem = leaf_cert.public_bytes(
        serialization.Encoding.PEM
    ).decode("ascii")
    key_pem = leaf_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return cert_pem, key_pem


def parse_certificate_serial(cert_pem: str) -> str:
    """Extract the OU attribute (== our cert_serial) from a PEM cert."""
    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    for attr in cert.subject:
        if attr.oid == NameOID.ORGANIZATIONAL_UNIT_NAME:
            return str(attr.value)
    raise ValueError("certificate has no OU attribute")


def parse_certificate_public_id(cert_pem: str) -> str:
    """Extract the CN attribute (== the attestation public_id)."""
    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    for attr in cert.subject:
        if attr.oid == NameOID.COMMON_NAME:
            return str(attr.value)
    raise ValueError("certificate has no CN attribute")


def verify_certificate_signature(
    cert_pem: str, *, backend: FakeHSMBackend | None = None
) -> bool:
    """Verify a leaf cert was signed by the (fake) V-Certified CA.

    Returns True iff the cert chains to our CA + the signature is
    valid. False otherwise (no exception raised on bad signature).
    """
    backend = backend or get_default_backend()
    try:
        cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    except ValueError:
        return False

    # 1. Issuer DN must match the CA subject.
    if cert.issuer != backend.ca_certificate.subject:
        return False

    # 2. Issuer must be the V-Certified CA (CN check).
    cn_match = False
    for attr in cert.issuer:
        if attr.oid == NameOID.COMMON_NAME and attr.value == CA_COMMON_NAME:
            cn_match = True
            break
    if not cn_match:
        return False

    # 3. Signature must verify under the CA public key.
    public_key = backend.ca_private_key.public_key()
    try:
        public_key.verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            padding.PKCS1v15(),
            cert.signature_hash_algorithm,
        )
    except InvalidSignature:
        return False
    return True


def _serial_from_str(s: str) -> int:
    """Map an opaque cert_serial string to a unique non-zero int.

    The cryptography API requires :class:`int` serials; the V-Certified
    serials we mint elsewhere are 16-byte hex strings prefixed with
    ``stub-`` (Phase 31.B legacy) or 32-byte hex (Phase 31.D issued).
    Strip the prefix + parse the hex tail; fall back to a hash if the
    tail isn't pure hex.
    """
    tail = s.removeprefix("stub-")
    try:
        return int(tail, 16) or 1
    except ValueError:
        # Deterministic hash → int for any opaque serial format.
        h = hashes.Hash(hashes.SHA256())
        h.update(s.encode("utf-8"))
        digest = h.finalize()
        n = int.from_bytes(digest[:16], "big")
        return n or 1


__all__ = [
    "issue_leaf_certificate",
    "parse_certificate_public_id",
    "parse_certificate_serial",
    "verify_certificate_signature",
]
