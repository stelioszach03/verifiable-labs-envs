"""Async-client tests — same coverage as sync, with `await`."""
from __future__ import annotations

import httpx
import pytest

from verifiable_labs import (
    AsyncClient,
    HealthStatus,
    LeaderboardResponse,
    SubmitResponse,
)


async def test_async_client_default_base_url():
    c = AsyncClient()
    assert c.base_url == "http://localhost:8000"
    await c.aclose()


async def test_async_health_returns_typed_model(async_client, mock_api, health_payload):
    mock_api.get("/v1/health").mock(return_value=httpx.Response(200, json=health_payload))
    h = await async_client.health()
    assert isinstance(h, HealthStatus)
    assert h.version == "0.1.0-alpha"
    await async_client.aclose()


async def test_async_environments(async_client, mock_api, envs_payload):
    mock_api.get("/v1/environments").mock(return_value=httpx.Response(200, json=envs_payload))
    envs = await async_client.environments()
    assert envs.count == 2
    await async_client.aclose()


async def test_async_evaluate_one_shot(
    async_client, mock_api, session_create_payload, submit_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    env = async_client.env("sparse-fourier-recovery")
    out = await env.evaluate(seed=42, answer="…")
    assert isinstance(out, SubmitResponse)
    assert out.reward == pytest.approx(0.842)
    await async_client.aclose()


async def test_async_session_submit_history(
    async_client, mock_api, session_create_payload, submit_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    env = async_client.env("sparse-fourier-recovery")
    session = await env.start_session(seed=42)
    assert session.complete is False
    await session.submit(answer_text="x")
    assert session.complete is True
    assert len(session.history) == 1
    await async_client.aclose()


async def test_async_leaderboard_top_models(
    async_client, mock_api, leaderboard_payload,
):
    mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(200, json=leaderboard_payload),
    )
    lb = await async_client.leaderboard("sparse-fourier-recovery")
    assert isinstance(lb, LeaderboardResponse)
    assert lb.top_models(n=1)[0].model == "anthropic/claude-haiku-4.5"
    await async_client.aclose()


async def test_async_context_manager_closes():
    async with AsyncClient(base_url="https://api.test.example") as client:
        assert client._http is not None
    # after exit; no re-use


async def test_async_404_raises_not_found(async_client, mock_api):
    mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(404, json={"detail": "no env"}),
    )
    from verifiable_labs import NotFoundError
    with pytest.raises(NotFoundError):
        await async_client.leaderboard("missing")
    await async_client.aclose()
