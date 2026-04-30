"""GET /v1/audit/{calibration_id} — metadata + eval history."""
from __future__ import annotations


async def test_audit_returns_metadata(client, api_key, calibrated) -> None:
    plaintext, _ = api_key
    r = await client.get(
        f"/v1/audit/{calibrated['calibration_id']}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["calibration_id"] == calibrated["calibration_id"]
    assert out["alpha"] == 0.1
    assert out["nonconformity"] == "scaled_residual"
    assert out["target_coverage"] == 0.9
    assert out["evaluations"] == []


async def test_audit_includes_evaluation_history(
    client, api_key, calibrated, gauss_traces
) -> None:
    plaintext, _ = api_key
    cal_id = calibrated["calibration_id"]

    # Run two evaluations on the same calibration.
    for seed in (10, 11):
        ev = await client.post(
            "/v1/evaluate",
            json={"calibration_id": cal_id, "traces": gauss_traces(20, seed=seed)},
            headers={"X-Vlabs-Key": plaintext},
        )
        assert ev.status_code == 200

    r = await client.get(
        f"/v1/audit/{cal_id}", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200
    out = r.json()
    assert len(out["evaluations"]) == 2
    assert all(e["n"] == 20 for e in out["evaluations"])


async def test_audit_unknown_id_404(client, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/audit/cal_00000000000000000000000000000000",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404


async def test_audit_invalid_id_404(client, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/audit/garbage_not_a_uuid", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
