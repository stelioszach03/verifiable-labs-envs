"""POST /v1/calibrate — happy path + every documented error code."""
from __future__ import annotations


async def test_calibrate_happy_path(client, api_key, gauss_traces) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": gauss_traces(200, seed=42),
        "metadata": {"experiment": "test-suite"},
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["calibration_id"].startswith("cal_")
    assert out["alpha"] == 0.1
    assert out["nonconformity"] == "scaled_residual"
    assert out["n_calibration"] == 200
    assert 0.0 < out["quantile"] < 10.0
    assert out["target_coverage"] == 0.9
    assert "mean" in out["nonconformity_stats"]


async def test_calibrate_invalid_alpha(client, api_key, gauss_traces) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.0,
        "nonconformity": "scaled_residual",
        "traces": gauss_traces(20),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422  # pydantic validation, alpha must be > 0


async def test_calibrate_traces_too_few(client, api_key) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": [{"predicted_reward": 0.5, "reference_reward": 0.5, "uncertainty": 0.1}],
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422  # pydantic min_length


async def test_calibrate_unknown_nonconformity(client, api_key, gauss_traces) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "not_a_real_score",
        "traces": gauss_traces(20),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 422  # pydantic Literal violation


async def test_calibrate_missing_uncertainty_for_scaled_residual(
    client, api_key
) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": [
            {"predicted_reward": 0.5, "reference_reward": 0.5},
            {"predicted_reward": 0.4, "reference_reward": 0.45},
        ],
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 400, r.text
    out = r.json()
    assert out["code"] == "missing_required_keys"


async def test_calibrate_abs_residual_does_not_need_uncertainty(
    client, api_key
) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "abs_residual",
        "traces": [
            {"predicted_reward": float(i) * 0.1, "reference_reward": float(i) * 0.1 + 0.05}
            for i in range(20)
        ],
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    assert r.json()["nonconformity"] == "abs_residual"


async def test_calibrate_persists_run(client, api_key, gauss_traces, session) -> None:
    from sqlalchemy import select

    from vlabs_api.db import CalibrationRun

    plaintext, info = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": gauss_traces(30, seed=7),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200

    # Fresh session — verify the row landed.
    from vlabs_api.db import _SessionFactory  # noqa: PLC0415

    async with _SessionFactory() as s:
        rows = (await s.execute(select(CalibrationRun))).scalars().all()
        assert len(rows) == 1
        assert rows[0].api_key_id == info["api_key_id"]
        assert rows[0].n_calibration == 30
