"""End-to-end session tests: create → submit → fetch."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _create(client: TestClient, env_id: str, seed: int = 0,
            env_kwargs: dict | None = None) -> dict:
    payload = {"env_id": env_id, "seed": seed}
    if env_kwargs is not None:
        payload["env_kwargs"] = env_kwargs
    r = client.post("/v1/sessions", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def test_create_session_with_qualified_env_id(client):
    body = _create(client, "stelioszach/sparse-fourier-recovery", seed=0)
    assert body["env_id"] == "sparse-fourier-recovery"
    assert "session_id" in body
    assert "observation" in body
    assert "prompt_text" in body["observation"]
    assert body["seed"] == 0


def test_create_session_with_bare_env_id(client):
    body = _create(client, "sparse-fourier-recovery", seed=1)
    assert body["env_id"] == "sparse-fourier-recovery"


def test_create_session_unknown_env_returns_404(client):
    r = client.post("/v1/sessions", json={"env_id": "no-such-env"})
    assert r.status_code == 404
    assert "Unknown env" in r.json()["detail"]


def test_create_session_default_seed_is_zero(client):
    body = _create(client, "sparse-fourier-recovery")
    assert body["seed"] == 0


def test_full_lifecycle_submit_truth_scores_high(client):
    # Use the calibration_quantile shortcut so session creation is instant.
    body = _create(
        client, "sparse-fourier-recovery", seed=0,
        env_kwargs={"calibration_quantile": 2.0},
    )
    session_id = body["session_id"]
    inputs = body["observation"]["inputs"]

    # Build a "truth" answer by reading the env's true support + amplitudes.
    # The API doesn't expose ground truth directly, but the env knows; we
    # sidestep by asking the env directly here for the test.
    from verifiable_labs_envs import load_environment
    env = load_environment("sparse-fourier-recovery", calibration_quantile=2.0)
    inst = env.generate_instance(seed=0)
    truth = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [
            int(round(float(v) * 1000)) for v in inst.x_true[inst.support_true]
        ],
    })

    r = client.post(
        f"/v1/sessions/{session_id}/submit",
        json={"answer_text": truth},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parse_ok"] is True
    assert body["complete"] is True  # single-turn env
    assert body["reward"] > 0.85, f"truth submission should score high: {body}"
    assert "nmse" in body["components"]
    assert isinstance(inputs, dict)


def test_submit_rejects_missing_answer(client):
    body = _create(client, "sparse-fourier-recovery")
    r = client.post(f"/v1/sessions/{body['session_id']}/submit", json={})
    assert r.status_code == 422
    assert "answer_text" in r.json()["detail"]


def test_submit_rejects_structured_answer_in_v01(client):
    body = _create(client, "sparse-fourier-recovery")
    r = client.post(
        f"/v1/sessions/{body['session_id']}/submit",
        json={"answer": {"x_hat": [0, 0, 0]}},
    )
    assert r.status_code == 422
    assert "v0.2" in r.json()["detail"]


def test_submit_invalid_json_returns_parse_failure(client):
    body = _create(client, "sparse-fourier-recovery")
    r = client.post(
        f"/v1/sessions/{body['session_id']}/submit",
        json={"answer_text": "not actually JSON"},
    )
    assert r.status_code == 200  # parse failure is a row, not a 4xx
    body = r.json()
    assert body["parse_ok"] is False
    assert body["reward"] == 0.0
    assert "parse_error" in body["meta"]


def test_submit_missing_session_returns_404(client):
    r = client.post(
        "/v1/sessions/00000000-0000-0000-0000-000000000000/submit",
        json={"answer_text": "{}"},
    )
    assert r.status_code == 404


def test_get_session_returns_state(client):
    create = _create(client, "sparse-fourier-recovery", seed=0,
                     env_kwargs={"calibration_quantile": 2.0})
    session_id = create["session_id"]
    # Submit a parse-failing answer so we have a recorded submission.
    client.post(
        f"/v1/sessions/{session_id}/submit",
        json={"answer_text": "garbage"},
    )
    r = client.get(f"/v1/sessions/{session_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == session_id
    assert body["env_id"] == "sparse-fourier-recovery"
    assert len(body["submissions"]) == 1
    assert body["submissions"][0]["parse_ok"] is False


def test_get_session_unknown_returns_404(client):
    r = client.get("/v1/sessions/not-a-real-uuid")
    assert r.status_code == 404


def test_multiturn_session_does_not_complete_after_one_submit(client):
    body = _create(
        client, "sparse-fourier-recovery-multiturn", seed=0,
        env_kwargs={"calibration_quantile": 2.0},
    )
    session_id = body["session_id"]
    from verifiable_labs_envs import load_environment
    env = load_environment("sparse-fourier-recovery-multiturn",
                           calibration_quantile=2.0)
    inst = env.generate_instance(seed=0)
    truth = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [
            int(round(float(v) * 1000)) for v in inst.x_true[inst.support_true]
        ],
    })
    r = client.post(
        f"/v1/sessions/{session_id}/submit",
        json={"answer_text": truth},
    )
    assert r.status_code == 200
    assert r.json()["complete"] is False  # multi-turn — server expects more turns
