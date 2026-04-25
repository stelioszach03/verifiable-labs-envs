"""Tests for ``Client`` (sync) basics: instantiation + health + envs."""
from __future__ import annotations

import httpx

from verifiable_labs import (
    Client,
    HealthStatus,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    __version__,
)


def test_version_is_alpha():
    assert __version__.startswith("0.1.0a"), __version__


def test_client_default_base_url():
    c = Client()
    assert c.base_url == "http://localhost:8000"
    c.close()


def test_client_explicit_base_url():
    c = Client(base_url="https://api.test.example/")
    # trailing slash stripped
    assert c.base_url == "https://api.test.example"
    c.close()


def test_client_context_manager_closes_http_client():
    with Client() as c:
        assert c._http is not None
    # No assertion on _http after close — just that the with-block
    # doesn't raise.


def test_health_returns_typed_model(client, mock_api, health_payload):
    mock_api.get("/v1/health").mock(return_value=httpx.Response(200, json=health_payload))
    h = client.health()
    assert isinstance(h, HealthStatus)
    assert h.version == "0.1.0-alpha"
    assert h.sessions_active == 3


def test_environments_returns_typed_list(client, mock_api, envs_payload):
    mock_api.get("/v1/environments").mock(return_value=httpx.Response(200, json=envs_payload))
    envs = client.environments()
    assert envs.count == 2
    assert envs.environments[0].id == "sparse-fourier-recovery"
    assert envs.environments[1].multi_turn is True


def test_404_raises_not_found(client, mock_api):
    mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(404, json={"detail": "Unknown env: xyz"}),
    )
    try:
        client.leaderboard("xyz")
    except NotFoundError as exc:
        assert exc.status_code == 404
        assert "Unknown env" in str(exc)
    else:
        raise AssertionError("expected NotFoundError")


def test_429_raises_rate_limit(client, mock_api):
    mock_api.get("/v1/health").mock(
        return_value=httpx.Response(
            429, json={"detail": "Rate limit exceeded: 30 per minute"},
        ),
    )
    try:
        client.health()
    except RateLimitError as exc:
        assert exc.status_code == 429
        assert "Rate limit" in str(exc)
    else:
        raise AssertionError("expected RateLimitError")


def test_500_raises_server_error(client, mock_api):
    mock_api.get("/v1/health").mock(return_value=httpx.Response(500, text="boom"))
    try:
        client.health()
    except ServerError as exc:
        assert exc.status_code == 500
    else:
        raise AssertionError("expected ServerError")


def test_transport_error_on_connection_refused(client, mock_api):
    mock_api.get("/v1/health").mock(side_effect=httpx.ConnectError("refused"))
    try:
        client.health()
    except TransportError as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("expected TransportError")


def test_user_agent_header_set(client, mock_api, health_payload):
    route = mock_api.get("/v1/health").mock(
        return_value=httpx.Response(200, json=health_payload),
    )
    client.health()
    request = route.calls[0].request
    assert "verifiable-labs-sdk" in request.headers.get("user-agent", "")


def test_api_key_header_added_when_set(mock_api, health_payload):
    route = mock_api.get("/v1/health").mock(
        return_value=httpx.Response(200, json=health_payload),
    )
    with Client(api_key="sk-test", base_url="https://api.test.example") as c:
        c.health()
    assert route.calls[0].request.headers.get("x-vl-api-key") == "sk-test"


def test_api_key_header_absent_when_unset(mock_api, health_payload):
    route = mock_api.get("/v1/health").mock(
        return_value=httpx.Response(200, json=health_payload),
    )
    with Client(base_url="https://api.test.example") as c:
        c.health()
    assert "x-vl-api-key" not in route.calls[0].request.headers


def test_inject_external_http_client_does_not_close_it():
    http = httpx.Client(base_url="https://api.test.example")
    c = Client(base_url="https://api.test.example", http_client=http)
    c.close()  # should not close http (we own it)
    assert not http.is_closed
    http.close()
