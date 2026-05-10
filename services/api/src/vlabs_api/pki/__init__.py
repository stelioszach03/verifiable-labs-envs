"""V-Certified PKI infrastructure (Phase 31.D).

Provides certificate authority + leaf-certificate issuance + CRL signing
for the V-Certified attestation programme. Each approved attestation
receives a unique X.509 leaf certificate signed by the V-Certified CA;
the cert is bundled with the attestation in the public registry.

Two backends:
- ``fake_hsm`` — in-memory CA keypair generated on first use; used by
  tests (gated on VLABS_LOCAL_FAKE_PKI=true) and local dev. Key
  material is non-persistent and never touches disk.
- ``kms_hsm`` (deferred to production hardening) — would back the CA
  private keys with AWS KMS or GCP KMS so the API process never sees
  unwrapped key material.

Production hardening (deferred):
- Two-tier hierarchy (root CA → intermediate CA → leaf), with the root
  kept offline.
- HSM-resident root key.
- 4096-bit RSA on root, 2048-bit on intermediate (cost / latency
  tradeoff).
- Manual CRL re-signing on a daily cadence.
"""
from __future__ import annotations

from vlabs_api.pki.cert_issuer import (
    issue_leaf_certificate,
    parse_certificate_serial,
    verify_certificate_signature,
)
from vlabs_api.pki.crl import build_crl_pem, parse_crl_serials
from vlabs_api.pki.fake_hsm import (
    FakeHSMBackend,
    get_default_backend,
    reset_default_backend_for_tests,
)

__all__ = [
    "FakeHSMBackend",
    "build_crl_pem",
    "get_default_backend",
    "issue_leaf_certificate",
    "parse_certificate_serial",
    "parse_crl_serials",
    "reset_default_backend_for_tests",
    "verify_certificate_signature",
]
