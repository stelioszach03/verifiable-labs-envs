"""POST /v1/evaluate — coverage report shape + persistence."""
from __future__ import annotations


async def test_evaluate_happy_path(client, api_key, calibrated, gauss_traces) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": calibrated["calibration_id"],
        "traces": gauss_traces(200, seed=99),
        "tolerance": 0.05,
    }
    r = await client.post(
        "/v1/evaluate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["calibration_id"] == calibrated["calibration_id"]
    assert out["n"] == 200
    assert 0.0 <= out["empirical_coverage"] <= 1.0
    assert out["target_coverage"] == 0.9
    assert isinstance(out["passes"], bool)
    assert out["tolerance"] == 0.05
    assert "mean" in out["nonconformity"]


async def test_evaluate_unknown_calibration(client, api_key, gauss_traces) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": "cal_00000000000000000000000000000000",
        "traces": gauss_traces(10),
    }
    r = await client.post(
        "/v1/evaluate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
    assert r.json()["code"] == "calibration_not_found"


async def test_evaluate_persists_evaluation(
    client, api_key, calibrated, gauss_traces
) -> None:
    from sqlalchemy import select

    from vlabs_api.db import Evaluation, _SessionFactory

    plaintext, _ = api_key
    body = {
        "calibration_id": calibrated["calibration_id"],
        "traces": gauss_traces(50, seed=11),
    }
    r = await client.post(
        "/v1/evaluate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200

    async with _SessionFactory() as s:
        rows = (await s.execute(select(Evaluation))).scalars().all()
        assert len(rows) == 1
        assert rows[0].n == 50
