"""Tier-aware rate limit — 100 RPM free, 1000 pro, 10000 team."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta


async def test_free_tier_blocks_at_100_rpm(client, api_key) -> None:
    """101 hits in tight succession on /v1/usage produces a 429 on the 101st."""
    plaintext, _ = api_key
    headers = {"X-Vlabs-Key": plaintext}
    for i in range(100):
        r = await client.get("/v1/usage", headers=headers)
        assert r.status_code == 200, f"hit {i} failed: {r.text}"
    r101 = await client.get("/v1/usage", headers=headers)
    assert r101.status_code == 429
    body = r101.json()
    assert body["code"] == "rate_limited"
    assert "free" in body["detail"]


async def test_pro_tier_allows_more_than_100(client, api_key, session) -> None:
    """An active 'pro' subscription lifts the cap to 1000/min."""
    from vlabs_api.db import Subscription

    plaintext, info = api_key
    now = datetime.now(UTC)
    session.add(
        Subscription(
            user_id=info["user_id"],
            stripe_subscription_id=f"sub_pro_test_{info['user_id'].hex[:8]}",
            tier="pro",
            status="active",
            current_period_start=now - timedelta(days=1),
            current_period_end=now + timedelta(days=29),
        )
    )
    await session.commit()

    headers = {"X-Vlabs-Key": plaintext}
    # Burst 150 calls — would 429 free, must succeed pro.
    for i in range(150):
        r = await client.get("/v1/usage", headers=headers)
        assert r.status_code == 200, f"pro tier hit {i} blocked: {r.text}"


async def test_rate_limit_isolated_per_key(client, api_key, session) -> None:
    """One key getting limited doesn't affect another key."""
    import uuid

    from vlabs_api.auth import generate_plaintext_key, hash_plaintext_key, key_prefix
    from vlabs_api.db import APIKey, User

    plaintext_a, _ = api_key

    # Burn through the quota for key A.
    for _ in range(100):
        await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext_a})

    # Key B is untouched.
    user_b = User(email=f"b-{uuid.uuid4().hex[:8]}@example.com")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    session.add(
        APIKey(
            user_id=user_b.id,
            key_hash=hash_plaintext_key(plaintext_b),
            key_prefix=key_prefix(plaintext_b),
            name="b",
        )
    )
    await session.commit()

    r = await client.get("/v1/usage", headers={"X-Vlabs-Key": plaintext_b})
    assert r.status_code == 200
