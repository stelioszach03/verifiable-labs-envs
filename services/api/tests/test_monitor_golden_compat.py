"""Golden-shape tests for the Phase 28 monitor endpoints.

Pins the JSON response shapes for ``/v1/monitors/*`` so a future
schema change has to be intentional + reviewed. Pattern mirrors
``test_backwards_compat.py``.
"""
from __future__ import annotations

from typing import Any

from httpx import AsyncClient


def _shape(value: Any, path: str = "") -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        out.add((path, "dict"))
        for k, v in value.items():
            sub = f"{path}.{k}" if path else k
            out |= _shape(v, sub)
    elif isinstance(value, list):
        out.add((path, "list"))
        if value:
            out |= _shape(value[0], f"{path}[]")
    elif value is None:
        out.add((path, "null"))
    else:
        out.add((path, type(value).__name__))
    return out


_PAYLOAD = {
    "name": "qwen-prod",
    "model_endpoint": "https://api.openai.com/v1",
    "model_name": "gpt-4o-mini",
    "auth_token": "sk-test-customer-key-XXXXXXXXXXXXXXXX",
    "cadence": "daily",
    "env_subset": ["math-algebra"],
    "episodes_per_env": 5,
}


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


# ── POST /v1/monitors create-shape ─────────────────────────────────


async def test_compat_post_monitors_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    r = await client.post("/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext))
    assert r.status_code == 201
    expected = {
        ("", "dict"),
        ("monitor_id", "str"),
        ("name", "str"),
        ("status", "str"),
        ("cadence", "str"),
        ("next_run_at", "str"),
        ("auth_token_fingerprint", "str"),
        ("projected_monthly_episodes", "int"),
        ("tier_limit_episodes", "int"),
        ("created_at", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── GET /v1/monitors list-shape ────────────────────────────────────


async def test_compat_get_monitors_list_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    await client.post("/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext))
    r = await client.get("/v1/monitors", headers=_hdr(plaintext))
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("items", "list"),
        ("total", "int"),
        ("limit", "int"),
        ("offset", "int"),
        ("items[]", "dict"),
        ("items[].monitor_id", "str"),
        ("items[].name", "str"),
        ("items[].model_name", "str"),
        ("items[].cadence", "str"),
        ("items[].status", "str"),
        ("items[].env_subset", "list"),
        ("items[].episodes_per_env", "int"),
        ("items[].next_run_at", "str"),
        ("items[].created_at", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── GET /v1/monitors/{id} detail-shape ─────────────────────────────


async def test_compat_get_monitor_detail_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    r = await client.get(
        f"/v1/monitors/{monitor_id}", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("monitor_id", "str"),
        ("name", "str"),
        ("model_endpoint", "str"),
        ("model_name", "str"),
        ("auth_token_fingerprint", "str"),
        ("cadence", "str"),
        ("env_subset", "list"),
        ("episodes_per_env", "int"),
        ("alert_channels", "list"),
        ("status", "str"),
        ("retention_days", "int"),
        ("created_at", "str"),
        ("updated_at", "str"),
        ("next_run_at", "str"),
        ("projected_monthly_episodes", "int"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── POST /v1/monitors/{id}/run trigger-shape ───────────────────────


async def test_compat_post_run_trigger_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    r = await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    assert r.status_code == 202
    expected = {
        ("", "dict"),
        ("monitor_run_id", "str"),
        ("monitor_id", "str"),
        ("scheduled_at", "str"),
        ("status", "str"),
        ("trigger", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── GET /v1/monitors/{id}/runs list-shape ──────────────────────────


async def test_compat_get_runs_list_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    create = await client.post(
        "/v1/monitors", json=_PAYLOAD, headers=_hdr(plaintext)
    )
    monitor_id = create.json()["monitor_id"]
    await client.post(
        f"/v1/monitors/{monitor_id}/run", headers=_hdr(plaintext)
    )
    r = await client.get(
        f"/v1/monitors/{monitor_id}/runs", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("items", "list"),
        ("total", "int"),
        ("limit", "int"),
        ("offset", "int"),
        ("items[]", "dict"),
        ("items[].monitor_run_id", "str"),
        ("items[].monitor_id", "str"),
        ("items[].scheduled_at", "str"),
        ("items[].status", "str"),
        ("items[].trigger", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── error-shape: 401 ───────────────────────────────────────────────


async def test_compat_post_monitors_unauth_shape(client: AsyncClient) -> None:
    r = await client.post("/v1/monitors", json=_PAYLOAD)
    assert r.status_code == 401
    expected = {
        ("", "dict"),
        ("type", "str"),
        ("title", "str"),
        ("status", "int"),
        ("code", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing


# ── monitor route registration ─────────────────────────────────────


def test_monitor_routes_present_in_app() -> None:
    """Mounted at /v1/monitors with the 5 expected paths (28.B + 28.C)."""
    from vlabs_api.main import app

    paths = sorted(r.path for r in app.router.routes)
    monitor_paths = [p for p in paths if p.startswith("/v1/monitors")]
    assert "/v1/monitors" in monitor_paths
    assert "/v1/monitors/{monitor_id}" in monitor_paths
    assert "/v1/monitors/{monitor_id}/run" in monitor_paths
    assert "/v1/monitors/{monitor_id}/runs" in monitor_paths
    assert "/v1/monitors/{monitor_id}/runs/{run_id}" in monitor_paths
