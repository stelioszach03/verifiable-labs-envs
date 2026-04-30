"""POST /v1/predict — interval shape + scope enforcement."""
from __future__ import annotations


async def test_predict_returns_interval(client, api_key, calibrated) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": calibrated["calibration_id"],
        "predicted_reward": 1.0,
        "uncertainty": 0.5,
    }
    r = await client.post(
        "/v1/predict", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["calibration_id"] == calibrated["calibration_id"]
    assert out["predicted_reward"] == 1.0
    assert out["sigma"] == 0.5
    lo, hi = out["interval"]
    assert lo < 1.0 < hi
    assert out["target_coverage"] == 0.9
    assert out["alpha"] == 0.1


async def test_predict_missing_sigma_for_scale_aware(client, api_key, calibrated) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": calibrated["calibration_id"],
        "predicted_reward": 1.0,
    }
    r = await client.post(
        "/v1/predict", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 400
    assert r.json()["code"] == "missing_required_keys"


async def test_predict_unknown_calibration_id(client, api_key) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": "cal_00000000000000000000000000000000",
        "predicted_reward": 1.0,
        "uncertainty": 0.5,
    }
    r = await client.post(
        "/v1/predict", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
    assert r.json()["code"] == "calibration_not_found"


async def test_predict_invalid_calibration_id(client, api_key) -> None:
    plaintext, _ = api_key
    body = {
        "calibration_id": "garbage_not_a_uuid",
        "predicted_reward": 1.0,
        "uncertainty": 0.5,
    }
    r = await client.post(
        "/v1/predict", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 404
    assert r.json()["code"] == "calibration_not_found"


async def test_predict_isolation_between_keys(client, api_key, calibrated, session) -> None:
    """A key cannot read another key's calibration_id (404 not 401)."""
    import uuid

    from vlabs_api.auth import generate_plaintext_key, hash_plaintext_key, key_prefix
    from vlabs_api.db import APIKey, User

    user2 = User(email=f"other-{uuid.uuid4().hex[:8]}@example.com")
    session.add(user2)
    await session.flush()
    pt2 = generate_plaintext_key()
    apik2 = APIKey(
        user_id=user2.id,
        key_hash=hash_plaintext_key(pt2),
        key_prefix=key_prefix(pt2),
        name="other",
    )
    session.add(apik2)
    await session.commit()

    body = {
        "calibration_id": calibrated["calibration_id"],
        "predicted_reward": 1.0,
        "uncertainty": 0.5,
    }
    r = await client.post("/v1/predict", json=body, headers={"X-Vlabs-Key": pt2})
    assert r.status_code == 404, r.text
