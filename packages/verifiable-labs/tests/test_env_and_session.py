"""Tests for ``Environment``, ``Session``, and ``Leaderboard`` flows."""
from __future__ import annotations

import httpx
import pytest

from verifiable_labs import (
    LeaderboardResponse,
    SessionState,
    SubmitResponse,
)


def test_evaluate_one_shot_returns_submit_response(
    client, mock_api, session_create_payload, submit_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    env = client.env("sparse-fourier-recovery")
    out = env.evaluate(seed=42, answer="…model output JSON…")
    assert isinstance(out, SubmitResponse)
    assert out.reward == pytest.approx(0.842)
    assert out.complete is True
    assert out.parse_ok is True


def test_evaluate_passes_qualified_id_through(
    client, mock_api, session_create_payload, submit_payload,
):
    route = mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    client.env("stelioszach/sparse-fourier-recovery").evaluate(seed=0, answer="x")
    body = route.calls[0].request.read().decode()
    assert "stelioszach/sparse-fourier-recovery" in body


def test_evaluate_forwards_env_kwargs(
    client, mock_api, session_create_payload, submit_payload,
):
    route = mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    client.env("sparse-fourier-recovery").evaluate(
        seed=5, answer="x", env_kwargs={"calibration_quantile": 2.0},
    )
    import json
    body = json.loads(route.calls[0].request.read().decode())
    assert body["seed"] == 5
    assert body["env_kwargs"] == {"calibration_quantile": 2.0}


def test_start_session_returns_session_with_observation(
    client, mock_api, session_create_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    session = client.env("sparse-fourier-recovery").start_session(seed=42)
    assert session.session_id == "s-123"
    assert session.seed == 42
    assert session.complete is False  # haven't submitted yet
    assert "prompt_text" in session.observation
    assert session.metadata["adapter_attached"] is True


def test_session_submit_appends_to_history(
    client, mock_api, session_create_payload, submit_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    session = client.env("sparse-fourier-recovery").start_session(seed=42)
    assert len(session.history) == 0
    result = session.submit(answer_text="…model output…")
    assert result.complete is True
    assert session.complete is True
    assert len(session.history) == 1


def test_session_submit_without_arguments_raises(
    client, mock_api, session_create_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    session = client.env("sparse-fourier-recovery").start_session(seed=42)
    with pytest.raises(ValueError, match="answer"):
        session.submit()


def test_session_refresh_resyncs_state(
    client, mock_api, session_create_payload, session_state_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.get("/v1/sessions/s-123").mock(
        return_value=httpx.Response(200, json=session_state_payload),
    )
    session = client.env("sparse-fourier-recovery").start_session(seed=42)
    state = session.refresh()
    assert isinstance(state, SessionState)
    assert state.complete is True
    assert session.complete is True
    assert len(session.history) == 1


def test_evaluate_with_dict_answer_uses_structured_path(
    client, mock_api, session_create_payload, submit_payload,
):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    submit_route = mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(200, json=submit_payload),
    )
    client.env("x").evaluate(seed=0, answer={"x_hat": [1, 2, 3]})
    import json
    body = json.loads(submit_route.calls[0].request.read().decode())
    assert "answer" in body
    assert body["answer"] == {"x_hat": [1, 2, 3]}
    assert "answer_text" not in body


# ── Leaderboard ─────────────────────────────────────────────────


def test_leaderboard_typed_response(client, mock_api, leaderboard_payload):
    mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(200, json=leaderboard_payload),
    )
    lb = client.leaderboard("sparse-fourier-recovery")
    assert isinstance(lb, LeaderboardResponse)
    assert lb.env_id == "sparse-fourier-recovery"
    assert len(lb.rows) == 3


def test_leaderboard_top_models_sort(client, mock_api, leaderboard_payload):
    mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(200, json=leaderboard_payload),
    )
    lb = client.leaderboard("sparse-fourier-recovery")
    top2 = lb.top_models(n=2)
    assert len(top2) == 2
    assert top2[0].model == "anthropic/claude-haiku-4.5"
    assert top2[1].model == "openai/gpt-5.4"


def test_leaderboard_passes_env_id_query(client, mock_api, leaderboard_payload):
    route = mock_api.get("/v1/leaderboard").mock(
        return_value=httpx.Response(200, json=leaderboard_payload),
    )
    client.leaderboard("foo")
    assert route.calls[0].request.url.params.get("env_id") == "foo"


def test_session_create_404_raises_not_found(client, mock_api):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(404, json={"detail": "Unknown env: xyz"}),
    )
    from verifiable_labs import NotFoundError
    with pytest.raises(NotFoundError):
        client.env("xyz").start_session(seed=0)


def test_submit_422_invalid_request(client, mock_api, session_create_payload):
    mock_api.post("/v1/sessions").mock(
        return_value=httpx.Response(201, json=session_create_payload),
    )
    mock_api.post("/v1/sessions/s-123/submit").mock(
        return_value=httpx.Response(
            422,
            json={"detail": "Provide either 'answer_text' or 'answer'."},
        ),
    )
    from verifiable_labs import InvalidRequestError
    session = client.env("x").start_session(seed=0)
    with pytest.raises(InvalidRequestError):
        # Not really empty — we send empty answer_text="" which the server treats as set,
        # and our mock returns 422 regardless. Good enough.
        session.submit(answer_text="")
