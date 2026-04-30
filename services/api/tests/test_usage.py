"""GET /v1/usage — counters + tier resolution + quota math."""
from __future__ import annotations

from datetime import UTC


async def test_usage_initial_state(client, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["tier"] == "free"
    assert out["quota"]["traces_per_month"] == 10_000
    assert out["quota"]["rpm"] == 100
    assert out["usage"]["traces"] == 0
    assert out["usage"]["calibrations"] == 0
    assert out["usage"]["evaluations"] == 0
    assert out["usage"]["predictions"] == 0
    assert out["remaining"]["traces"] == 10_000


async def test_usage_increments_after_calibrate(
    client, api_key, gauss_traces
) -> None:
    plaintext, _ = api_key
    body = {
        "alpha": 0.1,
        "nonconformity": "scaled_residual",
        "traces": gauss_traces(75, seed=3),
    }
    r = await client.post(
        "/v1/calibrate", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200

    r2 = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r2.status_code == 200
    out = r2.json()
    assert out["usage"]["traces"] == 75
    assert out["usage"]["calibrations"] == 1
    assert out["remaining"]["traces"] == 10_000 - 75


async def test_usage_predict_counts_one_trace(
    client, api_key, calibrated
) -> None:
    plaintext, _ = api_key

    # /v1/calibrate above already used 50 traces; now one /v1/predict call.
    r = await client.post(
        "/v1/predict",
        json={
            "calibration_id": calibrated["calibration_id"],
            "predicted_reward": 1.0,
            "uncertainty": 0.5,
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200

    r2 = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    out = r2.json()
    # 50 from calibrate fixture + 1 from predict
    assert out["usage"]["traces"] == 51
    assert out["usage"]["predictions"] == 1
    assert out["usage"]["calibrations"] == 1


async def test_usage_pro_tier_when_subscription_active(
    client, api_key, session
) -> None:
    """Insert an active 'pro' subscription; tier resolves to 'pro'."""
    from datetime import datetime, timedelta

    from vlabs_api.db import Subscription

    plaintext, info = api_key
    now = datetime.now(UTC)
    sub = Subscription(
        user_id=info["user_id"],
        stripe_subscription_id=f"sub_test_{info['user_id'].hex[:8]}",
        tier="pro",
        status="active",
        current_period_start=now - timedelta(days=1),
        current_period_end=now + timedelta(days=29),
    )
    session.add(sub)
    await session.commit()

    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext})
    assert r.status_code == 200
    out = r.json()
    assert out["tier"] == "pro"
    assert out["quota"]["traces_per_month"] == 1_000_000
    assert out["quota"]["rpm"] == 1_000
