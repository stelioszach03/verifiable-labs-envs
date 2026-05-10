"""Phase 31.E — standards crosswalks unit + endpoint tests."""
from __future__ import annotations

from httpx import AsyncClient

from vlabs_api.standards import (
    CROSSWALK_VERSION,
    KNOWN_FRAMEWORKS,
    get_crosswalk,
    list_all_crosswalks,
)

# ── module-level invariants ────────────────────────────────────────


def test_crosswalk_version_matches_locked_value() -> None:
    assert CROSSWALK_VERSION == "v0.0.1"


def test_known_frameworks_match_attestation_allowed_standards() -> None:
    from vlabs_api.attestation_service import ALLOWED_STANDARDS

    assert set(KNOWN_FRAMEWORKS) == ALLOWED_STANDARDS


def test_list_all_crosswalks_returns_one_entry_per_framework() -> None:
    out = list_all_crosswalks()
    assert set(out.keys()) == set(KNOWN_FRAMEWORKS)
    for fw, crosswalk in out.items():
        assert len(crosswalk) > 0, f"{fw} has empty crosswalk"


def test_each_crosswalk_entry_has_evidence_kinds() -> None:
    for fw in KNOWN_FRAMEWORKS:
        for entry in get_crosswalk(fw):
            assert len(entry.evidence_kinds) > 0
            for kind in entry.evidence_kinds:
                # All evidence kinds must be a valid D9 artifact kind.
                assert kind in {
                    "training_doc",
                    "audit_report",
                    "monitor_record",
                    "rm_record",
                    "prm_record",
                    "change_mgmt",
                    "legal_signoff",
                    "third_party_audit",
                }


def test_each_crosswalk_entry_has_framework_clauses() -> None:
    for fw in KNOWN_FRAMEWORKS:
        for entry in get_crosswalk(fw):
            assert len(entry.framework_clauses) > 0


def test_vc_1_1_present_in_every_framework() -> None:
    """VC-1.1 (training-data provenance) is the unifying control."""
    for fw in KNOWN_FRAMEWORKS:
        ids = {e.vc_control_id for e in get_crosswalk(fw)}
        assert "VC-1.1" in ids, f"{fw} is missing VC-1.1"


# ── GET /v1/standards ──────────────────────────────────────────────


async def test_list_standards_endpoint(client: AsyncClient) -> None:
    r = await client.get("/v1/standards")
    assert r.status_code == 200
    body = r.json()
    assert set(body["frameworks"]) == set(KNOWN_FRAMEWORKS)
    assert body["crosswalk_version"] == CROSSWALK_VERSION


async def test_list_standards_endpoint_no_auth(client: AsyncClient) -> None:
    """The standards endpoint is intentionally public (verifiers don't
    have Vlabs API keys)."""
    r = await client.get("/v1/standards")
    assert r.status_code == 200


# ── GET /v1/standards/{framework} ─────────────────────────────────


async def test_get_iso_42001_endpoint(client: AsyncClient) -> None:
    r = await client.get("/v1/standards/iso_42001")
    assert r.status_code == 200
    body = r.json()
    assert body["framework"] == "iso_42001"
    assert body["crosswalk_version"] == CROSSWALK_VERSION
    assert len(body["entries"]) > 0


async def test_get_nist_ai_rmf_endpoint(client: AsyncClient) -> None:
    r = await client.get("/v1/standards/nist_ai_rmf")
    assert r.status_code == 200
    body = r.json()
    assert body["framework"] == "nist_ai_rmf"
    # NIST RMF has Govern subcategory.
    clauses = {
        c
        for entry in body["entries"]
        for c in entry["framework_clauses"]
    }
    assert any(c.startswith("GOVERN") for c in clauses)


async def test_get_eu_ai_act_endpoint(client: AsyncClient) -> None:
    r = await client.get("/v1/standards/eu_ai_act")
    assert r.status_code == 200
    body = r.json()
    clauses = {
        c
        for entry in body["entries"]
        for c in entry["framework_clauses"]
    }
    # EU AI Act references articles + annexes.
    assert any("Article" in c for c in clauses)


async def test_get_soc2_endpoint(client: AsyncClient) -> None:
    r = await client.get("/v1/standards/soc2")
    assert r.status_code == 200
    body = r.json()
    clauses = {
        c
        for entry in body["entries"]
        for c in entry["framework_clauses"]
    }
    # SOC 2 references CC criteria.
    assert any(c.startswith("CC") for c in clauses)


async def test_get_unknown_framework_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.get("/v1/standards/nope")
    assert r.status_code == 404


# ── attestation issuance picks up the crosswalk version ────────────


async def test_attestation_issuance_records_crosswalk_version(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/attestations",
        headers={"X-Vlabs-Key": plaintext},
        json={
            "organization": "ACME",
            "scope_type": "model",
            "scope_subject": "x",
            "tier": "bronze",
            "cycle": "annual",
            "standards_requested": ["iso_42001", "soc2"],
        },
    )
    assert r.status_code == 201
    body = r.json()
    alignment = body["standards_alignment"]
    assert alignment["crosswalk_version"] == CROSSWALK_VERSION
    assert alignment["framework_versions"]["iso_42001"].startswith(
        "ISO/IEC 42001"
    )
    assert (
        alignment["framework_versions"]["soc2"].startswith("TSC 2017")
    )
