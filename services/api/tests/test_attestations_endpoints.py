"""Integration tests for the Phase 31.B owner attestation endpoints.

Exercises the 7 owner surfaces: create / list / detail / patch /
artifact upload / renew / revoke. Multi-party admin approval lives
in 31.E lifecycle integration tests.
"""
from __future__ import annotations

import base64
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _bronze_payload(**overrides) -> dict:
    base = {
        "organization": "ACME AI Corp",
        "scope_type": "model",
        "scope_subject": "acme-llm-v1",
        "tier": "bronze",
        "cycle": "annual",
        "standards_requested": ["iso_42001", "soc2"],
    }
    base.update(overrides)
    return base


def _gold_payload(**overrides) -> dict:
    base = {
        "organization": "Frontier Lab",
        "scope_type": "organization",
        "scope_subject": "frontier-lab-prod",
        "tier": "gold",
        "cycle": "continuous",
        "standards_requested": ["iso_42001", "nist_ai_rmf", "eu_ai_act"],
    }
    base.update(overrides)
    return base


# ── POST /v1/attestations ──────────────────────────────────────────


async def test_create_basic_bronze(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("att_")
    assert body["public_id"].startswith("vl-")
    assert body["status"] == "draft"
    assert body["tier"] == "bronze"
    assert body["cycle"] == "annual"
    assert body["organization"] == "ACME AI Corp"
    assert body["artifact_count"] == 0
    assert body["standards_alignment"]["standards"] == ["iso_42001", "soc2"]
    assert body["cert_serial"] is None
    assert body["issued_at"] is None


async def test_create_gold_continuous(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_gold_payload()
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["tier"] == "gold"
    assert body["cycle"] == "continuous"


async def test_create_rejects_tier_cycle_mismatch(
    client: AsyncClient, api_key
) -> None:
    """D3-D / D4-B mapping: Bronze must use annual, Gold must use
    continuous. Mismatched combos return 409."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json=_bronze_payload(cycle="continuous"),
    )
    assert r.status_code == 409
    assert r.json()["code"] == "attestation_invalid_state"


async def test_create_rejects_unknown_standard(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers=_hdr(plaintext),
        json={
            "organization": "X",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": ["bogus_framework"],
        },
    )
    # Pydantic Literal validation catches it before reaching the service.
    assert r.status_code == 422


async def test_create_requires_auth(client: AsyncClient) -> None:
    r = await client.post("/v1/attestations", json=_bronze_payload())
    assert r.status_code == 401


# ── GET /v1/attestations ───────────────────────────────────────────


async def test_list_empty(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/attestations", headers=_hdr(plaintext))
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_list_returns_owner_rows_only(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    for org in ("Org A", "Org B", "Org C"):
        await client.post(
            "/v1/attestations",
            headers=_hdr(plaintext),
            json=_bronze_payload(organization=org),
        )
    r = await client.get("/v1/attestations", headers=_hdr(plaintext))
    body = r.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    assert {item["organization"] for item in body["items"]} == {
        "Org A",
        "Org B",
        "Org C",
    }


async def test_list_filter_by_tier(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_gold_payload()
    )
    r = await client.get(
        "/v1/attestations",
        headers=_hdr(plaintext),
        params={"tier": "gold"},
    )
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["tier"] == "gold"


async def test_list_filter_by_status(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = r1.json()["id"]
    # Withdraw it.
    await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"action": "withdraw"},
    )
    r = await client.get(
        "/v1/attestations",
        headers=_hdr(plaintext),
        params={"status": "withdrawn"},
    )
    assert r.json()["total"] == 1


async def test_list_pagination(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    for i in range(5):
        await client.post(
            "/v1/attestations",
            headers=_hdr(plaintext),
            json=_bronze_payload(scope_subject=f"model-{i}"),
        )
    r = await client.get(
        "/v1/attestations",
        headers=_hdr(plaintext),
        params={"limit": 2, "offset": 1},
    )
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5


async def test_list_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/v1/attestations")
    assert r.status_code == 401


# ── GET /v1/attestations/{id} ──────────────────────────────────────


async def test_get_detail(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.get(
        f"/v1/attestations/{att_id}", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == att_id
    assert body["status"] == "draft"


async def test_get_detail_404_unknown(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    bogus = f"att_{uuid.uuid4().hex}"
    r = await client.get(
        f"/v1/attestations/{bogus}", headers=_hdr(plaintext)
    )
    assert r.status_code == 404
    assert r.json()["code"] == "attestation_not_found"


async def test_get_detail_404_for_other_owner(
    client: AsyncClient, api_key
) -> None:
    """Another customer's attestation surfaces as 404, not 403, to
    avoid leaking existence (mirrors Phase 22+ posture)."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]

    # Create a second user via a separate test helper.

    from vlabs_api import db
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    from vlabs_api.db import APIKey, User

    async with db._SessionFactory() as s:  # type: ignore[misc]
        other = User(email=f"other-{uuid.uuid4().hex[:8]}@example.com")
        s.add(other)
        await s.flush()
        other_plaintext = generate_plaintext_key()
        s.add(
            APIKey(
                user_id=other.id,
                key_hash=hash_plaintext_key(other_plaintext),
                key_prefix=key_prefix(other_plaintext),
                name="other-key",
            )
        )
        await s.commit()

    r = await client.get(
        f"/v1/attestations/{att_id}",
        headers={"X-Vlabs-Key": other_plaintext},
    )
    assert r.status_code == 404


async def test_get_detail_invalid_id_shape(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/attestations/not-a-valid-id", headers=_hdr(plaintext)
    )
    assert r.status_code == 404


# ── PATCH /v1/attestations/{id} ────────────────────────────────────


async def test_patch_metadata_in_draft(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"organization": "ACME (renamed)"},
    )
    assert r.status_code == 200
    assert r.json()["organization"] == "ACME (renamed)"


async def test_patch_metadata_blocked_after_submit(
    client: AsyncClient, api_key
) -> None:
    """Once submitted, metadata is locked. R3 mitigation."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    # Upload required artifacts so submit can succeed.
    for kind in ("training_doc", "audit_report", "legal_signoff"):
        await client.post(
            f"/v1/attestations/{att_id}/artifacts",
            headers=_hdr(plaintext),
            json={
                "kind": kind,
                "filename": f"{kind}.pdf",
                "content_b64": _b64(b"fake artifact bytes"),
                "encrypted": False,
            },
        )
    submit = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert submit.status_code == 200
    assert submit.json()["status"] == "submitted"

    # Now metadata edit must be blocked.
    r = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"organization": "ACME (renamed)"},
    )
    assert r.status_code == 409
    assert r.json()["code"] == "attestation_invalid_state"


async def test_patch_submit_missing_required_artifacts_409(
    client: AsyncClient, api_key
) -> None:
    """Bronze tier requires training_doc + audit_report +
    legal_signoff. Missing any → 409."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    # Upload only one required artifact.
    await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "doc.pdf",
            "content_b64": _b64(b"x"),
            "encrypted": False,
        },
    )
    r = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"action": "submit"},
    )
    assert r.status_code == 409
    assert "missing required artifacts" in r.json()["detail"]


async def test_patch_withdraw_from_draft(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"action": "withdraw"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "withdrawn"


async def test_patch_unknown_action_409(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.patch(
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"action": "approve"},  # rejected by Pydantic Literal
    )
    assert r.status_code == 422


# ── POST /v1/attestations/{id}/artifacts ───────────────────────────


async def test_artifact_upload_basic(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "training.pdf",
            "content_b64": _b64(b"PDF-1.4 fake content"),
            "encrypted": False,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("attart_")
    assert body["attestation_id"] == att_id
    assert body["kind"] == "training_doc"
    assert body["sha256_hash"] and len(body["sha256_hash"]) == 64
    assert body["size_bytes"] == len(b"PDF-1.4 fake content")
    assert body["encrypted"] is False
    assert body["storage_uri"].startswith("r2://vlabs-attestations/")
    assert "training.pdf" in body["storage_uri"]


async def test_artifact_upload_encrypted_flag(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "change_mgmt",
            "filename": "internal.pdf",
            "content_b64": _b64(b"sensitive"),
            "encrypted": True,
        },
    )
    assert r.status_code == 201
    assert r.json()["encrypted"] is True


async def test_artifact_upload_too_large_rejected(
    client: AsyncClient, api_key
) -> None:
    """50 MB cap per D9. Fake the size by sending a base64 block
    representing >50 MB decoded."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]

    big = b"\x00" * (50 * 1024 * 1024 + 1)
    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "big.bin",
            "content_b64": _b64(big),
            "encrypted": False,
        },
    )
    assert r.status_code == 413
    assert r.json()["code"] == "attestation_artifact_too_large"


async def test_artifact_upload_empty_content_rejected(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "x.pdf",
            "content_b64": _b64(b""),
            "encrypted": False,
        },
    )
    # Pydantic min_length=1 catches it on the way in.
    assert r.status_code == 422


async def test_artifact_upload_invalid_base64(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "x.pdf",
            "content_b64": "not-valid-b64!!!",
            "encrypted": False,
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "attestation_invalid_artifact"


async def test_artifact_upload_blocked_after_approval(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Approved attestations have immutable artifacts. The admin path
    that reaches 'approved' lives in 31.E lifecycle tests; here we
    fake the status directly via the ORM to exercise the gate."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]

    # Cheat: flip status to 'approved' directly via a fresh session
    # (the autouse session fixture's transaction state isn't visible
    # to the live ASGI test client, so we use a separate session).
    from vlabs_api import db as _db
    from vlabs_api.db import Attestation
    from vlabs_api.ids import parse_attestation_id

    async with _db._SessionFactory() as s:  # type: ignore[misc]
        from sqlalchemy import select

        row = (
            await s.execute(
                select(Attestation).where(
                    Attestation.id == parse_attestation_id(att_id)
                )
            )
        ).scalar_one()
        row.status = "approved"
        await s.commit()

    r = await client.post(
        f"/v1/attestations/{att_id}/artifacts",
        headers=_hdr(plaintext),
        json={
            "kind": "training_doc",
            "filename": "x.pdf",
            "content_b64": _b64(b"x"),
            "encrypted": False,
        },
    )
    assert r.status_code == 409


# ── POST /v1/attestations/{id}/renew ───────────────────────────────


async def test_renew_blocked_when_not_approved(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.post(
        f"/v1/attestations/{att_id}/renew",
        headers=_hdr(plaintext),
        json={},
    )
    assert r.status_code == 409
    assert r.json()["code"] == "attestation_invalid_state"


async def test_renew_idempotent_on_key(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Two POSTs with the same idempotency_key return the same renewal
    row within the 24 h window."""
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]

    # Force-approve via the ORM (admin path lands in 31.E).
    from sqlalchemy import select

    from vlabs_api import db as _db
    from vlabs_api.db import Attestation
    from vlabs_api.ids import parse_attestation_id

    async with _db._SessionFactory() as s:  # type: ignore[misc]
        row = (
            await s.execute(
                select(Attestation).where(
                    Attestation.id == parse_attestation_id(att_id)
                )
            )
        ).scalar_one()
        row.status = "approved"
        from datetime import UTC, datetime

        row.issued_at = datetime.now(UTC)
        row.cert_serial = "stub-test-serial"
        await s.commit()

    body = {"idempotency_key": "client-renewal-001"}
    r1 = await client.post(
        f"/v1/attestations/{att_id}/renew",
        headers=_hdr(plaintext),
        json=body,
    )
    assert r1.status_code == 201
    r2 = await client.post(
        f"/v1/attestations/{att_id}/renew",
        headers=_hdr(plaintext),
        json=body,
    )
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


# ── DELETE /v1/attestations/{id} ───────────────────────────────────


async def test_revoke_records_audit_row(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.request(
        "DELETE",
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "customer requested withdrawal"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "revoked"
    assert body["revoked_at"] is not None
    assert body["revocation_reason"] == "customer requested withdrawal"

    # Audit row must exist.
    from sqlalchemy import select

    from vlabs_api import db as _db
    from vlabs_api.db import AttestationAudit

    async with _db._SessionFactory() as s:  # type: ignore[misc]
        rows = (
            await s.execute(select(AttestationAudit))
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].decision == "revoke"
    assert rows[0].auditor_kind == "self"


async def test_revoke_blocked_in_terminal_state(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    # First revocation succeeds.
    r1 = await client.request(
        "DELETE",
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "first"},
    )
    assert r1.status_code == 200
    # Second revocation rejected — already in terminal state.
    r2 = await client.request(
        "DELETE",
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={"revocation_reason": "second"},
    )
    assert r2.status_code == 409


async def test_revoke_requires_reason(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/attestations", headers=_hdr(plaintext), json=_bronze_payload()
    )
    att_id = create.json()["id"]
    r = await client.request(
        "DELETE",
        f"/v1/attestations/{att_id}",
        headers=_hdr(plaintext),
        json={},
    )
    assert r.status_code == 422


# ── ID helpers ─────────────────────────────────────────────────────


def test_encode_attestation_id_round_trip() -> None:
    from vlabs_api.ids import encode_attestation_id, parse_attestation_id

    rid = uuid.uuid4()
    encoded = encode_attestation_id(rid)
    assert encoded.startswith("att_")
    assert len(encoded) == 4 + 32
    assert parse_attestation_id(encoded) == rid


def test_parse_attestation_id_invalid_raises() -> None:
    from vlabs_api.errors import AttestationNotFound
    from vlabs_api.ids import parse_attestation_id

    with pytest.raises(AttestationNotFound):
        parse_attestation_id("definitely-not-a-uuid")


def test_public_id_encoding_round_trip() -> None:
    from vlabs_api.ids import (
        encode_attestation_public_id,
        parse_attestation_public_id,
    )

    rid = uuid.uuid4()
    encoded = encode_attestation_public_id(rid)
    assert encoded.startswith("vl-")
    assert len(encoded) == 3 + 8
    bare = parse_attestation_public_id(encoded)
    assert len(bare) == 8
    # Same UUID always maps to same public_id (deterministic).
    assert encode_attestation_public_id(rid) == encoded


def test_public_id_excludes_ambiguous_chars() -> None:
    """Crockford alphabet omits I, L, O, U so codes are
    transcription-safe."""
    from vlabs_api.ids import encode_attestation_public_id

    for _ in range(50):
        encoded = encode_attestation_public_id(uuid.uuid4())
        bare = encoded[3:]
        for ch in "ILOU":
            assert ch not in bare


def test_parse_public_id_rejects_wrong_length() -> None:
    from vlabs_api.errors import AttestationNotFound
    from vlabs_api.ids import parse_attestation_public_id

    with pytest.raises(AttestationNotFound):
        parse_attestation_public_id("vl-TOOSHORT")
    with pytest.raises(AttestationNotFound):
        parse_attestation_public_id("vl-WAYTOOLONG12345")


def test_parse_public_id_rejects_non_crockford_chars() -> None:
    from vlabs_api.errors import AttestationNotFound
    from vlabs_api.ids import parse_attestation_public_id

    with pytest.raises(AttestationNotFound):
        parse_attestation_public_id("vl-ABCDEFIL")  # I + L not in alphabet
