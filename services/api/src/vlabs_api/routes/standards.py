"""``/v1/standards/`` — public V-Certified crosswalk endpoints (Phase 31.E).

Two endpoints, both unauthenticated (verifiers + auditors are 3rd
parties without a Vlabs API key):

- ``GET /v1/standards`` — list all four supported frameworks.
- ``GET /v1/standards/{framework}`` — full crosswalk for one framework.

Crosswalks are static data (no DB, no I/O), so we don't need
per-IP rate limits beyond what Cloudflare's edge layer provides.
"""
from __future__ import annotations

from fastapi import APIRouter

from vlabs_api.errors import APIError
from vlabs_api.standards import (
    CROSSWALK_VERSION,
    KNOWN_FRAMEWORKS,
    get_crosswalk,
)

router = APIRouter(tags=["standards"])


class StandardsFrameworkUnknown(APIError):
    """404 for a framework outside the locked KNOWN_FRAMEWORKS subset."""

    status_code = 404
    code = "standards_framework_unknown"
    title = "framework is not in the V-Certified crosswalk set"


@router.get("/standards")
async def list_standards() -> dict:
    """List all supported framework names + the locked crosswalk version."""
    return {
        "frameworks": list(KNOWN_FRAMEWORKS),
        "crosswalk_version": CROSSWALK_VERSION,
    }


@router.get("/standards/{framework}")
async def get_standard(framework: str) -> dict:
    if framework not in KNOWN_FRAMEWORKS:
        raise StandardsFrameworkUnknown(detail=f"framework={framework!r}")
    crosswalk = get_crosswalk(framework)
    return {
        "framework": framework,
        "crosswalk_version": CROSSWALK_VERSION,
        "entries": [
            {
                "vc_control_id": e.vc_control_id,
                "vc_control_title": e.vc_control_title,
                "framework_clauses": list(e.framework_clauses),
                "evidence_kinds": list(e.evidence_kinds),
                "notes": e.notes,
            }
            for e in crosswalk
        ],
    }


__all__ = ["router", "StandardsFrameworkUnknown"]
