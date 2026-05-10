"""Phase 31.F — OpenAPI surface coverage check.

Pin every Phase 31.B-E endpoint into the FastAPI-generated OpenAPI
schema so that an accidental router-deregistration silently breaking
the V-Certified product would fail this test.
"""
from __future__ import annotations

from httpx import AsyncClient

# ── owner endpoints (31.B + 31.C) ─────────────────────────────────


async def test_openapi_lists_all_owner_attestation_endpoints(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/v1/attestations" in paths
    assert "post" in paths["/v1/attestations"]
    assert "get" in paths["/v1/attestations"]
    assert "/v1/attestations/{attestation_id}" in paths
    for method in ("get", "patch", "delete"):
        assert method in paths["/v1/attestations/{attestation_id}"], method
    assert (
        "/v1/attestations/{attestation_id}/artifacts" in paths
    )
    assert "/v1/attestations/{attestation_id}/renew" in paths


# ── public verification (31.D) ─────────────────────────────────────


async def test_openapi_lists_all_public_verification_endpoints(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    paths = r.json()["paths"]
    assert "/v1/attestations/registry" in paths
    assert "/v1/attestations/verify/{public_id}" in paths
    assert (
        "/v1/attestations/verify-by-cert/{cert_serial}" in paths
    )
    assert "/v1/attestations/badge/{public_id}.svg" in paths
    assert "/v1/attestations/crl.pem" in paths


# ── standards crosswalks (31.E) ────────────────────────────────────


async def test_openapi_lists_standards_endpoints(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    paths = r.json()["paths"]
    assert "/v1/standards" in paths
    assert "/v1/standards/{framework}" in paths


# ── admin review board (31.E) ──────────────────────────────────────


async def test_openapi_lists_admin_attestation_endpoints(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    paths = r.json()["paths"]
    assert "/v1/admin/attestations/{attestation_id}/decisions" in paths
    assert "/v1/admin/attestations/{attestation_id}/audit-trail" in paths


# ── tag coverage ──────────────────────────────────────────────────


async def test_openapi_attestation_tags_all_present(
    client: AsyncClient,
) -> None:
    """Every Phase 31 tag must appear at least once in the spec."""
    r = await client.get("/openapi.json")
    paths = r.json()["paths"]
    seen_tags: set[str] = set()
    for ops in paths.values():
        for spec in ops.values():
            for t in spec.get("tags", []):
                seen_tags.add(t)
    expected = {
        "attestations",  # 31.B owner router
        "attestations-public",  # 31.D public router
        "standards",  # 31.E standards router
        "admin-attestations",  # 31.E admin router
    }
    assert expected.issubset(seen_tags), (
        f"missing tags: {expected - seen_tags}"
    )


# ── component schemas ─────────────────────────────────────────────


async def test_openapi_exposes_attestation_schemas(
    client: AsyncClient,
) -> None:
    """Every public Pydantic model is in the OpenAPI components schema."""
    r = await client.get("/openapi.json")
    schemas = r.json()["components"]["schemas"]
    expected = {
        "AttestationInfo",
        "AttestationSummary",
        "AttestationList",
        "AttestationCreateRequest",
        "AttestationPatchRequest",
        "AttestationArtifactInfo",
        "AttestationArtifactRequest",
        "AttestationRenewalInfo",
        "AttestationRenewalRequest",
        "AttestationRevokeRequest",
        "AttestationStandardsAlignment",
        "AttestationPublicSummary",
        "AttestationPublicList",
        "AttestationPublicInfo",
        "AttestationPublicCertificate",
        "AttestationAdminDecisionRequest",
        "AttestationAuditEntry",
    }
    assert expected.issubset(schemas.keys()), (
        f"missing component schemas: {expected - schemas.keys()}"
    )


# ── status code coverage ──────────────────────────────────────────


async def test_openapi_attestation_post_returns_201(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    spec = r.json()["paths"]["/v1/attestations"]["post"]
    assert "201" in spec["responses"]


async def test_openapi_artifact_upload_returns_201(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    spec = r.json()["paths"][
        "/v1/attestations/{attestation_id}/artifacts"
    ]["post"]
    assert "201" in spec["responses"]


async def test_openapi_admin_decision_post_returns_201(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    spec = r.json()["paths"][
        "/v1/admin/attestations/{attestation_id}/decisions"
    ]["post"]
    assert "201" in spec["responses"]


# ── prior-phase smoke (regression: 31 didn't unregister anything) ─


async def test_openapi_still_includes_phase_28_30_endpoints(
    client: AsyncClient,
) -> None:
    """Sanity check: 31's router shuffles didn't accidentally
    deregister any prior-phase endpoint families."""
    r = await client.get("/openapi.json")
    paths = r.json()["paths"]
    # Phase 22 training plane
    assert any(p.startswith("/v1/score") for p in paths)
    # Phase 23 datasets
    assert any(p.startswith("/v1/datasets") for p in paths)
    # Phase 28 monitors
    assert any(p.startswith("/v1/monitors") for p in paths)
    # Phase 29 reward models
    assert any(p.startswith("/v1/reward-models") for p in paths)
    # Phase 30 process reward models
    assert any(p.startswith("/v1/process-reward-models") for p in paths)
    # Phase 24-25 management plane
    assert "/v1/keys" in paths
    assert any(p.startswith("/v1/billing") for p in paths)
    # Phase 24 admin
    assert "/v1/admin/dashboard" in paths


# ── crosswalk content sanity ──────────────────────────────────────


async def test_crosswalk_endpoints_return_consistent_versions(
    client: AsyncClient,
) -> None:
    """All four /v1/standards/{framework} responses report the same
    crosswalk_version as /v1/standards."""
    top = (await client.get("/v1/standards")).json()
    expected_version = top["crosswalk_version"]
    for fw in top["frameworks"]:
        body = (await client.get(f"/v1/standards/{fw}")).json()
        assert body["crosswalk_version"] == expected_version
