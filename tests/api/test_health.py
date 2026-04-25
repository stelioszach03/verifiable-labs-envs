"""Tests for /v1/health."""
from __future__ import annotations


def test_health_returns_200(client):
    r = client.get("/v1/health")
    assert r.status_code == 200


def test_health_payload_has_required_fields(client):
    r = client.get("/v1/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"].endswith("-alpha"), body["version"]
    assert isinstance(body["uptime_s"], float)
    assert body["uptime_s"] >= 0.0
    assert body["sessions_active"] == 0


def test_health_includes_alpha_label(client):
    """v0.1.0-alpha is part of the health response so consumers can
    tell they're not on a stable release."""
    r = client.get("/v1/health")
    assert "alpha" in r.json()["version"].lower()


def test_openapi_doc_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/v1/health" in r.json()["paths"]


def test_swagger_ui_served(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger-ui" in r.text.lower()
