"""Phase 31.D — public verification endpoint integration tests.

Five endpoints under ``/v1/attestations/`` exercised end-to-end:
- ``GET /attestations/registry`` (60 req/min/IP)
- ``GET /attestations/verify/{public_id}`` (300 req/min/IP)
- ``GET /attestations/verify-by-cert/{cert_serial}`` (60 req/min/IP)
- ``GET /attestations/badge/{public_id}.svg`` (600 req/min/IP)
- ``GET /attestations/crl.pem`` (60 req/min/IP)
"""
from __future__ import annotations

import base64
import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.db import Attestation
from vlabs_api.ids import parse_attestation_id
from vlabs_api.pki import verify_certificate_signature
from vlabs_api.pki.crl import verify_crl_signature


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


# ── helper: drive an attestation to "approved" via the service ────


async def _approve_one(
    client: AsyncClient,
    plaintext: str,
    session: AsyncSession,
    *,
    organization: str = "ACME",
    scope_subject: str = "acme-llm-v1",
    tier: str = "bronze",
    cycle: str = "annual",
    standards: list[str] | None = None,
) -> Attestation:
    payload = {
        "organization": organization,
        "scope_type": "model",
        "scope_subject": scope_subject,
        "tier": tier,
        "cycle": cycle,
        "standards_requested": standards or ["iso_42001"],
    }
    r = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=payload
    )
    assert r.status_code == 201, r.text
    body = r.json()
    aid = body["id"]
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        upload = await client.post(
            f"/v1/attestations/{aid}/artifacts",
            headers=_hdr(plaintext),
            json={
                "kind": kind,
                "filename": f"{kind}.pdf",
                "content_b64": _b64(b"evidence"),
            },
        )
        assert upload.status_code == 201, upload.text

    r2 = await client.patch(
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r2.status_code == 200, r2.text

    uid = parse_attestation_id(aid)
    await svc.record_audit_decision(
        session,
        attestation_id=uid,
        auditor_kind="vlabs",
        auditor_user_id=None,
        auditor_label="vlabs",
        audit_summary={},
        decision="approve",
    )
    await session.commit()
    return (
        await session.execute(
            select(Attestation).where(Attestation.id == uid)
        )
    ).scalar_one()


# ── GET /v1/attestations/registry ──────────────────────────────────


async def test_registry_lists_only_public_statuses(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    # Approved attestation -> should appear in the registry.
    approved = await _approve_one(client, plaintext, session)
    # Draft attestation -> must NOT appear.
    draft = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "Draft Corp",
            "scope_type": "model",
            "scope_subject": "draft-llm",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": [],
        },
    )
    assert draft.status_code == 201

    r = await client.get("/v1/attestations/registry")
    assert r.status_code == 200, r.text
    body = r.json()
    public_ids = {item["public_id"] for item in body["items"]}
    assert approved.public_id in public_ids
    # Draft attestation does NOT leak through the public registry.
    assert all(item["status"] != "draft" for item in body["items"])


async def test_registry_pagination(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    for i in range(3):
        await _approve_one(
            client,
            plaintext,
            session,
            scope_subject=f"acme-llm-{i}",
        )
    r = await client.get("/v1/attestations/registry?limit=2&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0


async def test_registry_filter_by_status_revoked(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    a = await _approve_one(client, plaintext, session)
    r = await client.request(
        "DELETE",
        f"/v1/attestations/{svc.encode_attestation_id(a.id) if hasattr(svc, 'encode_attestation_id') else _attestation_id_from_uuid(a.id)}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "model decommissioned"},
    )
    assert r.status_code == 200, r.text
    listing = await client.get(
        "/v1/attestations/registry?status=revoked"
    )
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == "revoked"


async def test_registry_does_not_require_auth(
    client: AsyncClient,
) -> None:
    """Registry is unauthenticated — no header at all should still 200."""
    r = await client.get("/v1/attestations/registry")
    assert r.status_code == 200


# ── GET /v1/attestations/verify/{public_id} ───────────────────────


async def test_verify_by_public_id_returns_full_record(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    approved = await _approve_one(client, plaintext, session)
    r = await client.get(
        f"/v1/attestations/verify/{approved.public_id}"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["public_id"] == approved.public_id
    assert body["status"] == "approved"
    assert body["cert_serial"] is not None
    assert body["certificate_pem"].startswith("-----BEGIN CERTIFICATE-----")


async def test_verify_returns_pem_chain_to_v_certified_ca(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    approved = await _approve_one(client, plaintext, session)
    r = await client.get(
        f"/v1/attestations/verify/{approved.public_id}"
    )
    cert_pem = r.json()["certificate_pem"]
    assert verify_certificate_signature(cert_pem) is True


async def test_verify_unknown_public_id_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.get("/v1/attestations/verify/vl-NOSUCH00")
    assert r.status_code == 404


async def test_verify_invalid_public_id_shape_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.get("/v1/attestations/verify/notavalidid")
    assert r.status_code in (400, 404, 422)


async def test_verify_does_not_leak_drafts(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "Draft Co",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": [],
        },
    )
    assert r.status_code == 201
    pid = r.json()["public_id"]
    verify = await client.get(f"/v1/attestations/verify/{pid}")
    assert verify.status_code == 404


# ── GET /v1/attestations/verify-by-cert/{cert_serial} ──────────────


async def test_verify_by_cert_returns_pem(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    approved = await _approve_one(client, plaintext, session)
    r = await client.get(
        f"/v1/attestations/verify-by-cert/{approved.cert_serial}"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cert_serial"] == approved.cert_serial
    assert body["public_id"] == approved.public_id
    assert body["certificate_pem"].startswith("-----BEGIN CERTIFICATE-----")
    assert body["ca_certificate_pem"].startswith("-----BEGIN CERTIFICATE-----")
    assert body["attestation_status"] == "approved"


async def test_verify_by_cert_unknown_serial_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.get(
        "/v1/attestations/verify-by-cert/stub-nonexistent00"
    )
    assert r.status_code == 404


# ── GET /v1/attestations/badge/{public_id}.svg ────────────────────


async def test_badge_svg_contains_tier_and_status(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    approved = await _approve_one(client, plaintext, session, tier="bronze")
    r = await client.get(
        f"/v1/attestations/badge/{approved.public_id}.svg"
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert "V-Certified" in r.text
    assert "bronze" in r.text
    assert "approved" in r.text


async def test_badge_svg_unknown_id_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.get(
        "/v1/attestations/badge/vl-MISSING0.svg"
    )
    assert r.status_code == 404


async def test_badge_svg_uses_grey_color_for_revoked(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    # Bronze tier — _approve_one's stock 3 artifacts are sufficient.
    approved = await _approve_one(client, plaintext, session, tier="bronze")
    aid = (
        await session.execute(
            select(Attestation).where(Attestation.id == approved.id)
        )
    ).scalar_one()
    aid.status = "revoked"
    aid.revoked_at = approved.issued_at
    aid.revocation_reason = "test"
    await session.commit()

    r = await client.get(
        f"/v1/attestations/badge/{approved.public_id}.svg"
    )
    assert r.status_code == 200
    # Revoked badges use the neutral grey colour regardless of tier.
    assert "#999" in r.text


# ── GET /v1/attestations/crl.pem ───────────────────────────────────


async def test_crl_returns_pem_signed_by_ca(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    await _approve_one(client, plaintext, session)
    r = await client.get("/v1/attestations/crl.pem")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-pem-file")
    assert verify_crl_signature(r.text) is True


async def test_crl_includes_revoked_cert_serials(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    from vlabs_api.ids import encode_attestation_id

    plaintext, _ = api_key
    approved = await _approve_one(client, plaintext, session)
    aid = encode_attestation_id(approved.id)
    revoke = await client.request(
        "DELETE",
        f"/v1/attestations/{aid}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "key rotation"},
    )
    assert revoke.status_code == 200, revoke.text

    crl = await client.get("/v1/attestations/crl.pem")
    assert crl.status_code == 200
    from vlabs_api.pki import parse_crl_serials
    from vlabs_api.pki.cert_issuer import _serial_from_str

    serials = parse_crl_serials(crl.text)
    assert _serial_from_str(approved.cert_serial) in serials


# ── route disambiguation regression ────────────────────────────────


async def test_owner_attestation_id_path_still_routes_correctly(
    client: AsyncClient, api_key
) -> None:
    """Ensures registering the public router first didn't accidentally
    swallow the owner-side ``/attestations/{attestation_id}`` slot."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "Routing Test",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": [],
        },
    )
    assert r.status_code == 201
    aid = r.json()["id"]
    detail = await client.get(
        f"/v1/attestations/{aid}", headers=_hdr(plaintext)
    )
    assert detail.status_code == 200
    assert detail.json()["id"] == aid


# ── helper for tests using svc encoders directly ──────────────────


def _attestation_id_from_uuid(u: uuid.UUID) -> str:
    """Backstop until the test file imports encode_attestation_id."""
    from vlabs_api.ids import encode_attestation_id

    return encode_attestation_id(u)
