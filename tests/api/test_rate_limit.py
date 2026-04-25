"""Rate-limit + miscellaneous endpoint tests."""
from __future__ import annotations

from datetime import UTC

import pytest
from fastapi.testclient import TestClient

from verifiable_labs_api import create_app


def test_rate_limit_enforced_at_threshold():
    """A super-tight rate limit should trip 429 quickly."""
    app = create_app(rate_limit="3/minute", session_ttl_seconds=60)
    with TestClient(app) as client:
        codes = [client.get("/v1/health").status_code for _ in range(5)]
    # First three pass; subsequent ones must include at least one 429.
    assert codes[:3] == [200, 200, 200], codes
    assert 429 in codes[3:], codes


def test_default_rate_limit_does_not_trip_test_client():
    """At the default 30/min, two health calls should succeed."""
    app = create_app(session_ttl_seconds=60)
    with TestClient(app) as client:
        r1 = client.get("/v1/health")
        r2 = client.get("/v1/health")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_session_ttl_eviction(monkeypatch):
    """Sessions older than TTL are not retrievable."""
    from verifiable_labs_api.sessions import SessionStore
    store = SessionStore(ttl_seconds=1)
    # Construct a fake session
    session = store.make_session(
        env_id="x", seed=0, instance=None, env=None,
    )
    # Force expiration by mutating expires_at
    from datetime import datetime, timedelta
    session.expires_at = datetime.now(UTC) - timedelta(seconds=10)
    with pytest.raises(KeyError):
        store.get(session.session_id)


def test_create_session_invalid_env_kwargs_returns_400(client):
    r = client.post("/v1/sessions", json={
        "env_id": "sparse-fourier-recovery",
        "seed": 0,
        "env_kwargs": {"definitely_not_a_real_kwarg": 42},
    })
    assert r.status_code in (400, 422), r.text


def test_session_store_ttl_validation():
    from verifiable_labs_api.sessions import SessionStore
    with pytest.raises(ValueError, match="ttl_seconds"):
        SessionStore(ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds"):
        SessionStore(ttl_seconds=-1)
