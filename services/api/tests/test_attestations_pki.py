"""Phase 31.D — PKI module unit tests (fake-HSM backend).

Direct coverage for :mod:`vlabs_api.pki`:
- CA self-signed cert generation;
- leaf-cert issuance signed by the CA;
- signature verification chain;
- CRL build + parse + signature verification.

Real-HSM (kms_hsm) coverage is deferred to production hardening.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509

from vlabs_api.pki import (
    build_crl_pem,
    issue_leaf_certificate,
    parse_certificate_serial,
    parse_crl_serials,
    verify_certificate_signature,
)
from vlabs_api.pki.cert_issuer import (
    _serial_from_str,
    parse_certificate_public_id,
)
from vlabs_api.pki.crl import verify_crl_signature
from vlabs_api.pki.fake_hsm import (
    CA_COMMON_NAME,
    FakeHSMBackend,
    get_default_backend,
    reset_default_backend_for_tests,
)

# ── fake HSM backend ───────────────────────────────────────────────


def test_get_default_backend_returns_singleton() -> None:
    a = get_default_backend()
    b = get_default_backend()
    assert a is b


def test_reset_default_backend_returns_fresh_instance() -> None:
    a = get_default_backend()
    reset_default_backend_for_tests()
    b = get_default_backend()
    assert a is not b
    # Restore for downstream tests.
    reset_default_backend_for_tests()
    get_default_backend()


def test_fake_hsm_ca_certificate_is_self_signed() -> None:
    backend = FakeHSMBackend()
    assert backend.ca_certificate.subject == backend.ca_certificate.issuer


def test_fake_hsm_ca_certificate_pem_round_trip() -> None:
    backend = FakeHSMBackend()
    pem = backend.ca_certificate_pem
    cert = x509.load_pem_x509_certificate(pem.encode("ascii"))
    assert cert.subject == backend.ca_certificate.subject
    cn = next(
        attr.value for attr in cert.subject if attr.oid.dotted_string == "2.5.4.3"
    )
    assert cn == CA_COMMON_NAME


def test_ca_certificate_has_basic_constraints_ca_true() -> None:
    backend = FakeHSMBackend()
    bc = backend.ca_certificate.extensions.get_extension_for_class(
        x509.BasicConstraints
    )
    assert bc.value.ca is True
    assert bc.critical is True


def test_ca_certificate_validity_at_least_one_year() -> None:
    backend = FakeHSMBackend()
    not_before = backend.ca_certificate.not_valid_before
    not_after = backend.ca_certificate.not_valid_after
    assert (not_after - not_before) > timedelta(days=365)


# ── leaf cert issuance ─────────────────────────────────────────────


def test_issue_leaf_certificate_returns_pem_pair() -> None:
    cert_pem, key_pem = issue_leaf_certificate(
        public_id="vl-ABCD1234",
        organization="ACME AI Corp",
        cert_serial="stub-deadbeefcafe1234",
        lifetime=timedelta(days=365),
    )
    assert cert_pem.startswith("-----BEGIN CERTIFICATE-----")
    assert cert_pem.rstrip().endswith("-----END CERTIFICATE-----")
    assert key_pem.startswith("-----BEGIN PRIVATE KEY-----")
    assert key_pem.rstrip().endswith("-----END PRIVATE KEY-----")


def test_leaf_certificate_has_public_id_in_cn() -> None:
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-ABCD1234",
        organization="ACME",
        cert_serial="stub-1234567890abcdef",
        lifetime=timedelta(days=365),
    )
    assert parse_certificate_public_id(cert_pem) == "vl-ABCD1234"


def test_leaf_certificate_has_serial_in_ou() -> None:
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-XYZ12345",
        organization="ACME",
        cert_serial="stub-feedfacecafebeef",
        lifetime=timedelta(days=365),
    )
    assert parse_certificate_serial(cert_pem) == "stub-feedfacecafebeef"


def test_leaf_certificate_lifetime_matches_argument() -> None:
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-LIFETEST",
        organization="ACME",
        cert_serial="stub-aaaabbbbccccdddd",
        lifetime=timedelta(days=395),
    )
    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    span = cert.not_valid_after - cert.not_valid_before
    assert abs(span - timedelta(days=395)) < timedelta(minutes=2)


def test_leaf_certificate_basic_constraints_ca_false() -> None:
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-BASIC123",
        organization="ACME",
        cert_serial="stub-basic1234567890ab",
        lifetime=timedelta(days=365),
    )
    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
    assert bc.value.ca is False


def test_leaf_certificate_signed_by_ca_subject() -> None:
    backend = get_default_backend()
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-SIGNED12",
        organization="ACME",
        cert_serial="stub-signed1234567890",
        lifetime=timedelta(days=365),
    )
    cert = x509.load_pem_x509_certificate(cert_pem.encode("ascii"))
    assert cert.issuer == backend.ca_certificate.subject


# ── signature verification ─────────────────────────────────────────


def test_verify_certificate_signature_round_trip() -> None:
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-VERIFY1",
        organization="ACME",
        cert_serial="stub-verify12345abcdef",
        lifetime=timedelta(days=365),
    )
    assert verify_certificate_signature(cert_pem) is True


def test_verify_certificate_signature_rejects_garbage() -> None:
    assert verify_certificate_signature("not a pem") is False
    assert verify_certificate_signature("") is False


def test_verify_certificate_signature_rejects_other_ca() -> None:
    other_backend = FakeHSMBackend()
    cert_pem, _ = issue_leaf_certificate(
        public_id="vl-OTHERCA",
        organization="ACME",
        cert_serial="stub-otherca123456789a",
        lifetime=timedelta(days=365),
        backend=other_backend,
    )
    # Signed by OTHER backend; default backend should reject.
    assert verify_certificate_signature(cert_pem) is False


# ── CRL build + parse + verify ─────────────────────────────────────


def test_build_empty_crl_signed_by_ca() -> None:
    pem = build_crl_pem(revoked=[])
    assert pem.startswith("-----BEGIN X509 CRL-----")
    assert verify_crl_signature(pem) is True
    assert parse_crl_serials(pem) == []


def test_build_crl_with_entries_includes_serials() -> None:
    now = datetime.now(UTC)
    revoked = [
        ("stub-1234567890abcdef", now - timedelta(hours=1)),
        ("stub-cafebabedeadbeef", now - timedelta(minutes=5)),
    ]
    pem = build_crl_pem(revoked=revoked)
    serials = parse_crl_serials(pem)
    assert len(serials) == 2
    expected = {_serial_from_str(s) for s, _ in revoked}
    assert set(serials) == expected


def test_build_crl_uses_ca_signing_algorithm() -> None:
    pem = build_crl_pem(revoked=[])
    crl = x509.load_pem_x509_crl(pem.encode("ascii"))
    backend = get_default_backend()
    assert verify_crl_signature(pem) is True
    assert crl.issuer == backend.ca_certificate.subject


def test_serial_from_str_handles_stub_prefix() -> None:
    assert _serial_from_str("stub-deadbeef") == int("deadbeef", 16)


def test_serial_from_str_handles_pure_hex() -> None:
    assert _serial_from_str("0123456789abcdef") == int(
        "0123456789abcdef", 16
    )


def test_serial_from_str_handles_non_hex_via_hash() -> None:
    out = _serial_from_str("not-hex-at-all-nope")
    assert out > 0
    # Deterministic — same input twice gives same int.
    assert _serial_from_str("not-hex-at-all-nope") == out


def test_get_default_backend_refuses_in_prod_without_override(monkeypatch) -> None:
    """fake_hsm refuses to initialise when VLABS_ENVIRONMENT=prod and
    VLABS_LOCAL_FAKE_PKI is not set, to prevent accidental production
    use of the in-memory key material."""
    reset_default_backend_for_tests()
    monkeypatch.setenv("VLABS_ENVIRONMENT", "prod")
    monkeypatch.setenv("VLABS_LOCAL_FAKE_PKI", "")
    with pytest.raises(RuntimeError, match="kms_hsm"):
        get_default_backend()
    # Restore for downstream tests.
    monkeypatch.setenv("VLABS_LOCAL_FAKE_PKI", "true")
    monkeypatch.setenv("VLABS_ENVIRONMENT", "dev")
    reset_default_backend_for_tests()
    get_default_backend()
