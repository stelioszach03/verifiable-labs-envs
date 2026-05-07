"""Phase 22.E — full training-loop simulation tests.

Exercises the end-to-end training-API flow:

    1. POST /v1/instance        → fetch problem
    2. (customer trains locally; we substitute a simple oracle/zero solver)
    3. POST /v1/score           → submit completion, receive reward + audit_id
    4. GET  /v1/score/audit/{id} → verify the audit row matches

Repeated 100 times to catch counter-drift, idempotency-cache leaks,
adapter dispatch races, etc. Coverage in addition to per-endpoint
tests in 22.B/22.C/22.D.
"""
from __future__ import annotations

import json

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import AuditCall, UsageCounter


async def _instance_then_score(
    client: AsyncClient,
    plaintext: str,
    env_id: str,
    seed: int,
    *,
    use_oracle: bool = False,
) -> tuple[float, str]:
    """One training-loop tick. Returns (reward, audit_id).

    Resets the in-memory rate-limit bucket every call so long simulation
    loops don't hit the free-tier 100 RPM cap. Production rate-limiting
    is exercised in :mod:`test_ratelimit`.
    """
    from vlabs_api.ratelimit import reset_for_tests
    reset_for_tests()

    inst = await client.post(
        "/v1/instance",
        json={"env_id": env_id, "seed": seed},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert inst.status_code == 200, inst.text

    if use_oracle and env_id.startswith("math-algebra"):
        # Oracle path for symbolic-math envs: read gold from the env directly.
        from verifiable_labs_envs import load_environment
        env = load_environment(env_id, calibration_quantile=0.5)
        gold = env.generate_instance(seed=seed).gold_expr
        completion = json.dumps({"answer": gold, "confidence": 1.0})
    else:
        completion = ""

    score_r = await client.post(
        "/v1/score",
        json={"env_id": env_id, "seed": seed, "completion": completion},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert score_r.status_code == 200, score_r.text
    body = score_r.json()
    return body["reward"], body["audit_id"]


# ── full loop ─────────────────────────────────────────────────────


async def test_training_loop_100_iterations_no_drift(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Run the full instance→score loop 100x. Verify counter == 200
    (100 instance + 100 score) and every reward is in [0, 1]."""
    plaintext, info = api_key
    audit_ids: set[str] = set()
    rewards: list[float] = []
    for seed in range(100):
        reward, audit_id = await _instance_then_score(
            client, plaintext, "math-algebra", seed
        )
        audit_ids.add(audit_id)
        rewards.append(reward)

    assert len(audit_ids) == 100, "duplicate audit_ids — UPSERT race?"
    assert all(0.0 <= r <= 1.0 for r in rewards)

    res = await session.execute(select(UsageCounter))
    counter = res.scalar_one()
    # 100 /v1/instance calls + 100 /v1/score calls, all distinct seeds.
    assert counter.scores_count == 200


async def test_training_loop_oracle_solver_reaches_one(
    client: AsyncClient, api_key
) -> None:
    """Oracle solver (gold = answer) should score 1.0 on every iteration."""
    plaintext, _ = api_key
    rewards = []
    for seed in range(20):
        reward, _ = await _instance_then_score(
            client, plaintext, "math-algebra", seed, use_oracle=True
        )
        rewards.append(reward)
    assert all(r == 1.0 for r in rewards), f"oracle drift: {rewards}"


async def test_training_loop_audit_lookup_round_trip(
    client: AsyncClient, api_key
) -> None:
    """Each score response's audit_id resolves via /v1/score/audit/{id}."""
    plaintext, _ = api_key
    for seed in range(10):
        _, audit_id = await _instance_then_score(
            client, plaintext, "math-algebra", seed
        )
        r = await client.get(
            f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["audit_id"] == audit_id
        assert body["seed"] == seed
        assert body["env_id"] == "math-algebra"


async def test_training_loop_idempotency_holds_under_repeat(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """20 repeats of the SAME (idempotency_key, completion) → 1 audit row."""
    plaintext, _ = api_key
    body = {
        "env_id": "math-algebra",
        "seed": 0,
        "completion": "test",
        "idempotency_key": "training-loop-idempotency",
    }
    audit_ids = set()
    for _ in range(20):
        r = await client.post("/v1/score", json=body, headers={"X-Vlabs-Key": plaintext})
        assert r.status_code == 200
        audit_ids.add(r.json()["audit_id"])
    assert len(audit_ids) == 1, "idempotency cache leak"

    res = await session.execute(select(AuditCall))
    rows = res.scalars().all()
    assert len(rows) == 1


async def test_training_loop_pagination_accumulates_correctly(
    client: AsyncClient, api_key
) -> None:
    """50 score calls → /v1/score/audit?limit=20&offset=0 returns 20 newest;
    offset=40 returns 10."""
    plaintext, _ = api_key
    for seed in range(50):
        await _instance_then_score(client, plaintext, "math-algebra", seed)

    r1 = await client.get(
        "/v1/score/audit?limit=20", headers={"X-Vlabs-Key": plaintext}
    )
    body1 = r1.json()
    assert body1["total"] == 50
    assert len(body1["items"]) == 20

    r2 = await client.get(
        "/v1/score/audit?limit=20&offset=40", headers={"X-Vlabs-Key": plaintext}
    )
    body2 = r2.json()
    assert body2["total"] == 50
    assert len(body2["items"]) == 10


# ── multi-env loop ────────────────────────────────────────────────


async def test_training_loop_across_multiple_envs(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Cycle through 3 envs across 30 ticks; every env emits valid rewards."""
    plaintext, _ = api_key
    envs = ["math-algebra", "math-algebra-multiturn", "math-algebra-tools"]
    for i in range(30):
        env_id = envs[i % 3]
        reward, _ = await _instance_then_score(client, plaintext, env_id, seed=i)
        assert 0.0 <= reward <= 1.0

    res = await session.execute(select(AuditCall))
    rows = res.scalars().all()
    assert len(rows) == 30
    by_env = {env: 0 for env in envs}
    for row in rows:
        by_env[row.env_id] += 1
    assert all(v == 10 for v in by_env.values()), f"env imbalance: {by_env}"


# ── audit-row immutability under repeat ───────────────────────────


async def test_training_loop_audit_row_immutable_per_idempotency_key(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Repeated scoring with the same idempotency_key returns the original
    row even when subsequent completion text differs."""
    plaintext, _ = api_key

    body_1 = {
        "env_id": "math-algebra", "seed": 0, "completion": "first",
        "idempotency_key": "stable-key-77",
    }
    r1 = await client.post("/v1/score", json=body_1, headers={"X-Vlabs-Key": plaintext})
    assert r1.status_code == 200
    aud_id_1 = r1.json()["audit_id"]
    reward_1 = r1.json()["reward"]

    # 5 follow-ups with same key but different completions.
    for completion in ("second", "third", "fourth", "fifth", "sixth"):
        body = {**body_1, "completion": completion}
        r = await client.post("/v1/score", json=body, headers={"X-Vlabs-Key": plaintext})
        assert r.status_code == 200
        assert r.json()["audit_id"] == aud_id_1
        assert r.json()["reward"] == reward_1


# ── conformal interval consistency ────────────────────────────────


async def test_training_loop_conformal_interval_clamps_correctly(
    client: AsyncClient, api_key
) -> None:
    """Every conformal_interval is in [0, 1] and contains the reward."""
    plaintext, _ = api_key
    for seed in range(20):
        _, audit_id = await _instance_then_score(client, plaintext, "math-algebra", seed)
        r = await client.get(
            f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
        )
        body = r.json()
        low, high = body["conformal_interval"]
        assert 0.0 <= low <= high <= 1.0
        # The interval is centered on reward but clamped — reward may
        # not strictly be inside if it sits at the boundary.
        assert low <= body["reward"] + 1e-9
        assert body["reward"] - 1e-9 <= high


# ── env_version stability under repeat ────────────────────────────


async def test_training_loop_env_version_stable(
    client: AsyncClient, api_key
) -> None:
    """env_version on the first audit equals env_version on the 50th."""
    plaintext, _ = api_key
    versions = []
    for seed in range(50):
        _, audit_id = await _instance_then_score(client, plaintext, "math-algebra", seed)
        r = await client.get(
            f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
        )
        versions.append(r.json()["env_version"])
    assert len(set(versions)) == 1, f"env_version drift across calls: {set(versions)}"


# ── completion_hash determinism ───────────────────────────────────


async def test_training_loop_completion_hash_matches_local_sha(
    client: AsyncClient, api_key
) -> None:
    """Customer can re-derive their completion_hash locally and match."""
    import hashlib

    plaintext, _ = api_key
    completion = "deterministic test completion"
    expected_hash = hashlib.sha256(completion.encode("utf-8")).hexdigest()

    for seed in range(5):
        r = await client.post(
            "/v1/score",
            json={"env_id": "math-algebra", "seed": seed, "completion": completion},
            headers={"X-Vlabs-Key": plaintext},
        )
        audit_id = r.json()["audit_id"]
        a = await client.get(
            f"/v1/score/audit/{audit_id}", headers={"X-Vlabs-Key": plaintext}
        )
        assert a.json()["completion_hash"] == expected_hash
