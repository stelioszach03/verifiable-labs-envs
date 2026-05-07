"""Golden-shape tests for endpoints that pre-date Phase 22.

PHASE_22_PLAN.md §5.7 + §10 require that the following endpoint
contracts remain byte-identical pre- and post-Phase 22:

- ``GET /health`` (no ``/v1`` prefix; predates Phase 16 versioning)
- ``GET /v1/usage``
- ``GET /v1/audit/{calibration_id}`` (calibration audit, NOT score audit)
- ``POST /v1/calibrate``
- ``POST /v1/keys``

The 22.E commit lands the full per-endpoint snapshot suite. This
22.B scaffold pins the **shape skeletons** that 22.E expands:
each test asserts the response keys + types match the documented
schema. A future regression that drops a field or changes a type
fails here at the relevant sub-stage gate, not in production.

The pattern: walk the response JSON, reduce to ``(path, type)`` pairs,
and compare against a hand-coded golden tree. Type-only comparison
keeps the test stable across non-deterministic value fields
(timestamps, UUIDs, quantile floats).
"""
from __future__ import annotations

from typing import Any

from httpx import AsyncClient


def _shape(value: Any, path: str = "") -> set[tuple[str, str]]:
    """Reduce a JSON tree to ``{(path, type_name)}`` pairs."""
    out: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        out.add((path, "dict"))
        for k, v in value.items():
            sub = f"{path}.{k}" if path else k
            out |= _shape(v, sub)
    elif isinstance(value, list):
        out.add((path, "list"))
        # Probe one element if present — list elements share a shape.
        if value:
            out |= _shape(value[0], f"{path}[]")
    elif value is None:
        out.add((path, "null"))
    else:
        out.add((path, type(value).__name__))
    return out


# ── /v1/health ────────────────────────────────────────────────────


async def test_compat_health_shape(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    shape = _shape(r.json())
    expected = {
        ("", "dict"),
        ("status", "str"),
        ("version", "str"),
        ("environment", "str"),
    }
    missing = expected - shape
    assert not missing, f"missing keys after Phase 22: {missing}"


# ── /v1/usage ─────────────────────────────────────────────────────


async def test_compat_usage_shape(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200
    shape = _shape(r.json())
    expected = {
        ("", "dict"),
        ("tier", "str"),
        ("quota", "dict"),
        ("quota.traces_per_month", "int"),
        ("quota.rpm", "int"),
        ("current_period", "dict"),
        ("current_period.start", "str"),
        ("current_period.end", "str"),
        ("usage", "dict"),
        ("usage.traces", "int"),
        ("usage.calibrations", "int"),
        ("usage.evaluations", "int"),
        ("usage.predictions", "int"),
        ("remaining", "dict"),
        ("remaining.traces", "int"),
    }
    missing = expected - shape
    assert not missing, f"missing keys after Phase 22: {missing}"


# ── /v1/calibrate ─────────────────────────────────────────────────


async def test_compat_calibrate_shape(
    client: AsyncClient, api_key, gauss_traces
) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": gauss_traces(50, seed=1),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    shape = _shape(r.json())
    expected = {
        ("", "dict"),
        ("calibration_id", "str"),
        ("alpha", "float"),
        ("nonconformity", "str"),
        ("n_calibration", "int"),
        ("quantile", "float"),
        ("target_coverage", "float"),
        ("nonconformity_stats", "dict"),
        ("created_at", "str"),
    }
    missing = expected - shape
    assert not missing, f"missing keys after Phase 22: {missing}"


# ── /v1/audit/{calibration_id} ────────────────────────────────────


async def test_compat_calibration_audit_shape(
    client: AsyncClient, api_key, gauss_traces
) -> None:
    """Existing calibration audit (Phase 16) — must stay reachable.

    Distinct from the new ``/v1/score/audit/{audit_id}`` added in
    Phase 22.D (different path, different response shape).
    """
    plaintext, _ = api_key
    cal = await client.post(
        "/v1/calibrate",
        json={
            "alpha": 0.1,
            "nonconformity": "scaled_residual",
            "traces": gauss_traces(20, seed=0),
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    cal_id = cal.json()["calibration_id"]
    r = await client.get(
        f"/v1/audit/{cal_id}", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    shape = _shape(r.json())
    expected = {
        ("", "dict"),
        ("calibration_id", "str"),
        ("created_at", "str"),
        ("alpha", "float"),
        ("nonconformity", "str"),
        ("n_calibration", "int"),
        ("quantile", "float"),
        ("target_coverage", "float"),
        ("evaluations", "list"),
    }
    missing = expected - shape
    assert not missing, f"missing keys after Phase 22: {missing}"


# ── /v1/keys ─────────────────────────────────────────────────────


async def test_compat_keys_create_shape(
    client: AsyncClient, clerk_user, stub_clerk_verify
) -> None:
    """Existing key issuance (Phase 16 management plane) — Clerk JWT auth."""
    stub_clerk_verify()
    fake_jwt, _user = clerk_user
    r = await client.post(
        "/v1/keys",
        json={"name": "compat-test"},
        headers={"Authorization": f"Bearer {fake_jwt}"},
    )
    assert r.status_code == 200, r.text
    shape = _shape(r.json())
    expected = {
        ("", "dict"),
        ("id", "str"),
        ("prefix", "str"),
        ("name", "str"),
        ("created_at", "str"),
        ("plaintext_key", "str"),
    }
    missing = expected - shape
    assert not missing, f"missing keys after Phase 22: {missing}"
