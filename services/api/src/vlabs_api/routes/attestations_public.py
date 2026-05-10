"""``/v1/attestations/`` — public V-Certified verification endpoints (Phase 31.D).

Five endpoints, all unauthenticated, all IP-rate-limited:

- ``GET /v1/attestations/registry`` — paginated public list (60 req/min/IP).
- ``GET /v1/attestations/verify/{public_id}`` — single attestation by
  public_id (300 req/min/IP — verifiers may hit this in a tight loop).
- ``GET /v1/attestations/verify-by-cert/{cert_serial}`` — single
  attestation by issued cert serial (60 req/min/IP).
- ``GET /v1/attestations/badge/{public_id}.svg`` — embeddable badge
  (600 req/min/IP — public-web caching layers will absorb most of this).
- ``GET /v1/attestations/crl.pem`` — signed CRL of revoked certs
  (60 req/min/IP).

These endpoints intentionally bypass the ``X-Vlabs-Key`` middleware
because verifiers are 3rd-party tools that do not have a Vlabs API
key. Per-IP throttling is enforced via :func:`_enforce_ip_rate_limit`.
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api import attestation_service as svc
from vlabs_api.db import get_db
from vlabs_api.errors import APIError
from vlabs_api.ids import (
    parse_attestation_public_id,
)
from vlabs_api.pki import build_crl_pem, get_default_backend
from vlabs_api.ratelimit import _check_and_increment  # type: ignore[attr-defined]
from vlabs_api.schemas import (
    AttestationPublicCertificate,
    AttestationPublicInfo,
    AttestationPublicList,
    AttestationPublicSummary,
    AttestationStandardsAlignment,
)

router = APIRouter(tags=["attestations-public"])


# ── per-IP rate limit (Phase 31.D D8-C) ────────────────────────────


_PER_IP_RPM: dict[str, int] = {
    # endpoint key -> max requests per 60-second sliding window per IP.
    "registry": 60,
    "verify": 300,
    "verify-by-cert": 60,
    "badge": 600,
    "crl": 60,
}


class PublicEndpointRateLimited(APIError):
    """429 Too Many Requests for a public endpoint hit too often per IP."""

    status_code = 429
    code = "public_endpoint_rate_limited"
    title = "verification endpoint rate limit exceeded for this IP"


def _client_ip(request: Request) -> str:
    """Pull the verifier's IP from the request scope.

    Behind a CF / Fly proxy we trust the inbound connection's
    ``request.client.host`` because the platform terminates TLS and
    rewrites the source IP. ``X-Forwarded-For`` is NOT consulted here
    because spoofable headers from public clients would let an attacker
    bypass per-IP throttling.
    """
    if request.client is None:  # pragma: no cover — ASGI invariant
        return "unknown"
    return request.client.host or "unknown"


async def _enforce_ip_rate_limit(
    request: Request, *, key: str
) -> None:
    """Apply the existing :func:`_check_and_increment` bucket keyed on
    the verifier's IP + the endpoint name. Window is the same 60 s
    sliding window as the data-plane tiered limit; only RPM varies."""
    rpm = _PER_IP_RPM.get(key)
    if rpm is None:
        return
    bucket_key = f"vc-public:{key}:{_client_ip(request)}"
    ok, retry_after = await _check_and_increment(bucket_key, rpm)
    if not ok:
        raise PublicEndpointRateLimited(
            detail=f"limit={rpm}/min for endpoint={key}",
            retry_after=int(retry_after) + 1,
        )


# ── helpers ────────────────────────────────────────────────────────


def _attestation_to_public_info(
    row, *, certificate_pem: str | None = None
) -> AttestationPublicInfo:
    alignment_payload = row.standards_alignment or {}
    return AttestationPublicInfo(
        public_id=row.public_id,
        organization=row.organization,
        scope_type=row.scope_type,
        scope_subject=row.scope_subject,
        tier=row.tier,
        status=row.status,
        cycle=row.cycle,
        issued_at=row.issued_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        revocation_reason=row.revocation_reason,
        cert_serial=row.cert_serial,
        certificate_pem=certificate_pem,
        standards_alignment=AttestationStandardsAlignment(
            standards=alignment_payload.get("standards", []) or [],
            crosswalk_version=alignment_payload.get("crosswalk_version"),
            framework_versions=alignment_payload.get(
                "framework_versions", {}
            )
            or {},
        ),
    )


def _attestation_to_public_summary(row) -> AttestationPublicSummary:
    return AttestationPublicSummary(
        public_id=row.public_id,
        organization=row.organization,
        scope_type=row.scope_type,
        scope_subject=row.scope_subject,
        tier=row.tier,
        status=row.status,
        issued_at=row.issued_at,
        expires_at=row.expires_at,
    )


# ── GET /v1/attestations/registry ──────────────────────────────────


@router.get(
    "/attestations/registry", response_model=AttestationPublicList
)
async def public_registry(
    request: Request,
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
) -> AttestationPublicList:
    await _enforce_ip_rate_limit(request, key="registry")
    rows, total = await svc.list_for_public_registry(
        session, limit=limit, offset=offset, status=status
    )
    return AttestationPublicList(
        items=[_attestation_to_public_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# ── GET /v1/attestations/verify/{public_id} ───────────────────────


@router.get(
    "/attestations/verify/{public_id}",
    response_model=AttestationPublicInfo,
)
async def public_verify_by_id(
    public_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> AttestationPublicInfo:
    await _enforce_ip_rate_limit(request, key="verify")
    # Validate the input shape early; bad public_ids 404 instead of 500.
    parse_attestation_public_id(public_id)
    row = await svc.get_by_public_id(session, public_id=public_id)
    cert_pem: str | None = None
    if row.cert_serial:
        cert_row = await svc.get_certificate(
            session, cert_serial=row.cert_serial
        )
        cert_pem = cert_row.certificate_pem if cert_row else None
    return _attestation_to_public_info(row, certificate_pem=cert_pem)


# ── GET /v1/attestations/verify-by-cert/{cert_serial} ──────────────


@router.get(
    "/attestations/verify-by-cert/{cert_serial}",
    response_model=AttestationPublicCertificate,
)
async def public_verify_by_cert(
    cert_serial: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> AttestationPublicCertificate:
    await _enforce_ip_rate_limit(request, key="verify-by-cert")
    row = await svc.get_by_cert_serial(session, cert_serial=cert_serial)
    cert_row = await svc.get_certificate(session, cert_serial=cert_serial)
    return AttestationPublicCertificate(
        public_id=row.public_id,
        cert_serial=cert_serial,
        certificate_pem=cert_row.certificate_pem if cert_row else None,
        ca_certificate_pem=get_default_backend().ca_certificate_pem,
        issued_at=cert_row.issued_at if cert_row else None,
        revoked_at=cert_row.revoked_at if cert_row else None,
        attestation_status=row.status,
    )


# ── GET /v1/attestations/badge/{public_id}.svg ────────────────────


_BADGE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="186" height="20" role="img" aria-label="V-Certified: {tier}">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="186" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="80" height="20" fill="#555"/>
    <rect x="80" width="106" height="20" fill="{color}"/>
    <rect width="186" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text x="400" y="140" transform="scale(.1)" textLength="700">V-Certified</text>
    <text x="1330" y="140" transform="scale(.1)" textLength="960">{tier}: {status}</text>
  </g>
</svg>"""


_TIER_COLORS: dict[str, str] = {
    "bronze": "#cd7f32",
    "silver": "#a0a0a0",
    "gold": "#dfa500",
}


@router.get("/attestations/badge/{public_id}.svg")
async def public_badge_svg(
    public_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> Response:
    await _enforce_ip_rate_limit(request, key="badge")
    parse_attestation_public_id(public_id)
    row = await svc.get_by_public_id(session, public_id=public_id)
    color = _TIER_COLORS.get(row.tier, "#555")
    if row.status != "approved":
        color = "#999"
    body = _BADGE_TEMPLATE.format(
        tier=row.tier, status=row.status, color=color
    )
    return Response(
        content=body,
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "public, max-age=300",
        },
    )


# ── GET /v1/attestations/crl.pem ───────────────────────────────────


@router.get("/attestations/crl.pem")
async def public_crl(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> Response:
    await _enforce_ip_rate_limit(request, key="crl")
    revoked_certs = await svc.list_revoked_certs(session)
    pairs = [
        (c.cert_serial, c.revoked_at or datetime.now(UTC))
        for c in revoked_certs
    ]
    pem = build_crl_pem(revoked=pairs)
    return Response(
        content=pem,
        media_type="application/x-pem-file",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": 'inline; filename="v-certified.crl.pem"',
        },
    )


__all__ = ["router", "PublicEndpointRateLimited"]
