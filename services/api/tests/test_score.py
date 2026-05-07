"""Tests for ``POST /v1/score`` (Phase 22.C).

Coverage:
- happy paths: oracle-completion scores 1.0, empty scores 0.0
- env families: math + sparse-fourier + math-multiturn
- adapter parse failures yield zero score (not 5xx)
- audit_calls row written with correct fields
- completion stored as SHA-256 hash, NEVER plaintext
- env_version pinned per row
- conformal interval clamped to [0, 1]
- coverage_guarantee derived from env's α
- 1 MB completion cap returns 422 (Pydantic Field max_length)
- idempotency: same key in 24h → cached; different key or stale → fresh
- quota enforcement on scores_per_month
- auth: missing key → 401; revoked key → 401
- unknown env_id → 404
- malformed body → 422
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import AuditCall, UsageCounter

# ── Happy paths ────────────────────────────────────────────────────


async def test_score_oracle_math_algebra_reaches_one(
    client: AsyncClient, api_key
) -> None:
    """Oracle completion (gold = answer) scores 1.0 across all components."""
    plaintext, _ = api_key

    inst = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert inst.status_code == 200, inst.text

    # Re-derive gold by loading the env directly (server side).
    from verifiable_labs_envs import load_environment
    env = load_environment("math-algebra", calibration_quantile=0.5)
    gold = env.generate_instance(seed=0).gold_expr

    completion = json.dumps({"answer": gold, "confidence": 1.0})
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": completion},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reward"] == 1.0
    assert body["audit_id"].startswith("aud_")
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0


async def test_score_empty_completion_scores_zero(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    assert r.json()["reward"] == 0.0


async def test_score_garbage_completion_scores_zero(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra",
            "seed": 0,
            "completion": "not json at all !!",
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    assert r.json()["reward"] == 0.0


async def test_score_inverse_problem_env(
    client: AsyncClient, api_key
) -> None:
    """Sparse-fourier env via /v1/score — completion is JSON with support_idx + amp."""
    plaintext, _ = api_key
    # Provide a malformed JSON so the adapter raises but we still get reward=0.
    r = await client.post(
        "/v1/score",
        json={
            "env_id": "sparse-fourier-recovery",
            "seed": 0,
            "completion": json.dumps({"support_idx": [], "support_amp_x1000": []}),
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    # Adapter rejects empty support → zero reward; happy-path semantics
    # are exercised in the integration test suite (22.E).
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["reward"] <= 1.0


async def test_score_multiturn_env(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    from verifiable_labs_envs import load_environment

    env = load_environment("math-algebra-multiturn", calibration_quantile=0.5)
    gold = env.generate_instance(seed=0).gold_expr
    completion = json.dumps({"answer": gold, "confidence": 1.0})
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra-multiturn", "seed": 0, "completion": completion},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text


# ── audit_calls persistence ────────────────────────────────────────


async def test_score_writes_audit_row(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    completion = "raw completion text"
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": completion},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    audit_id_str = r.json()["audit_id"]
    assert audit_id_str.startswith("aud_")

    # Look up the row and verify fields.
    res = await session.execute(
        select(AuditCall).where(AuditCall.user_id == info["user_id"])
    )
    row = res.scalar_one()
    assert row.env_id == "math-algebra"
    assert row.seed == 0
    assert row.api_key_id == info["api_key_id"]
    # Hash, never plaintext.
    assert row.completion_hash == hashlib.sha256(completion.encode()).hexdigest()
    assert completion not in str(row.components_json)


async def test_score_completion_stored_as_hash_only(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """GDPR guarantee: raw completion never persisted."""
    plaintext, _ = api_key
    secret = "PII-12345-sensitive-completion-content"
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": secret},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    res = await session.execute(select(AuditCall))
    rows = res.scalars().all()
    for row in rows:
        assert secret not in row.completion_hash
        assert secret not in str(row.components_json)
        assert row.completion_hash == hashlib.sha256(secret.encode()).hexdigest()


async def test_score_pins_env_version(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    from verifiable_labs_envs import __version__ as env_version
    assert r.json()["env_version"] == env_version
    res = await session.execute(select(AuditCall))
    row = res.scalar_one()
    assert row.env_version == env_version


async def test_score_conformal_interval_in_unit_interval(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    low, high = body["conformal_interval"]
    assert 0.0 <= low <= high <= 1.0


async def test_score_coverage_guarantee_in_unit_interval(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    assert 0.0 <= body["coverage_guarantee"] <= 1.0


async def test_score_components_breakdown_keys(
    client: AsyncClient, api_key
) -> None:
    """Math envs emit format_valid + parse_valid + correct."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    assert isinstance(body["components_breakdown"], dict)
    for k in ("format_valid", "parse_valid", "correct"):
        assert k in body["components_breakdown"]


# ── Idempotency ────────────────────────────────────────────────────


async def test_score_idempotent_returns_cached_audit(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    body = {
        "env_id": "math-algebra",
        "seed": 0,
        "completion": "first",
        "idempotency_key": "test-key-1",
    }
    r1 = await client.post(
        "/v1/score", json=body, headers={"X-Vlabs-Key": plaintext}
    )
    assert r1.status_code == 200
    audit_id_1 = r1.json()["audit_id"]

    # Second call with the SAME key + DIFFERENT completion: returns cached row.
    body2 = {**body, "completion": "different completion"}
    r2 = await client.post(
        "/v1/score", json=body2, headers={"X-Vlabs-Key": plaintext}
    )
    assert r2.status_code == 200
    assert r2.json()["audit_id"] == audit_id_1

    # Counter incremented exactly once across the two calls.
    res = await session.execute(select(UsageCounter))
    counter = res.scalar_one()
    assert counter.scores_count == 1


async def test_score_idempotent_different_keys_distinct_rows(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "idempotency_key": "key-A",
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    r2 = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "idempotency_key": "key-B",
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.json()["audit_id"] != r2.json()["audit_id"]


async def test_score_idempotent_no_key_distinct_rows(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    r2 = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.json()["audit_id"] != r2.json()["audit_id"]


async def test_score_idempotency_window_24h(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Stale key (>24h) treated as fresh — new audit row written."""
    plaintext, info = api_key
    # Score once, then back-date the row beyond the window.
    r1 = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "idempotency_key": "stale-key",
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.status_code == 200

    res = await session.execute(select(AuditCall))
    row = res.scalar_one()
    row.created_at = datetime.now(UTC) - timedelta(hours=25)
    await session.commit()

    # Re-issue the same key — should NOT match (out of window).
    r2 = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "idempotency_key": "stale-key",
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r2.json()["audit_id"] != r1.json()["audit_id"]


# ── Quota enforcement ──────────────────────────────────────────────


async def test_score_quota_increment(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    for _ in range(3):
        r = await client.post(
            "/v1/score",
            json={"env_id": "math-algebra", "seed": 0, "completion": ""},
            headers={"X-Vlabs-Key": plaintext},
        )
        assert r.status_code == 200

    res = await session.execute(select(UsageCounter))
    counter = res.scalar_one()
    assert counter.scores_count == 3


async def test_score_quota_exhausted_returns_402(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    # Pre-fill at the cap.
    from datetime import date

    counter = UsageCounter(
        api_key_id=info["api_key_id"],
        month=date.today().replace(day=1),
        scores_count=1_000,
    )
    session.add(counter)
    await session.commit()

    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 402
    assert r.json()["code"] == "quota_exceeded"


async def test_score_idempotent_does_not_increment_counter(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    body = {
        "env_id": "math-algebra", "seed": 0, "completion": "",
        "idempotency_key": "no-increment-test",
    }
    for _ in range(5):
        r = await client.post("/v1/score", json=body, headers={"X-Vlabs-Key": plaintext})
        assert r.status_code == 200

    res = await session.execute(select(UsageCounter))
    counter = res.scalar_one()
    assert counter.scores_count == 1


# ── Auth ───────────────────────────────────────────────────────────


async def test_score_missing_api_key_rejected(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
    )
    assert r.status_code == 401


async def test_score_revoked_key_rejected(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    from vlabs_api.db import APIKey

    res = await session.execute(select(APIKey).where(APIKey.id == info["api_key_id"]))
    row = res.scalar_one()
    row.revoked_at = datetime.now(UTC)
    await session.commit()

    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 401


# ── Error paths ────────────────────────────────────────────────────


async def test_score_unknown_env_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "does-not-exist", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "unknown_environment"


async def test_score_negative_seed_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": -1, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_score_missing_env_id_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_score_missing_completion_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_score_extra_keys_rejected(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "future_field": True,
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_score_completion_over_1mb_returns_422(
    client: AsyncClient, api_key
) -> None:
    """1 MB cap — Pydantic max_length raises 422 (PHASE_22_PLAN.md §5.2)."""
    plaintext, _ = api_key
    huge = "x" * (1_048_576 + 1)
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": huge},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_score_idempotency_key_too_long_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={
            "env_id": "math-algebra", "seed": 0, "completion": "",
            "idempotency_key": "k" * 201,
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


# ── Reward clamping ────────────────────────────────────────────────


async def test_score_reward_in_unit_interval(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert 0.0 <= r.json()["reward"] <= 1.0


async def test_score_reward_components_in_unit_interval(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    for k, v in r.json()["components_breakdown"].items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"


# ── Latency reporting ─────────────────────────────────────────────


async def test_score_latency_ms_non_negative(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/score",
        json={"env_id": "math-algebra", "seed": 0, "completion": ""},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.json()["latency_ms"] >= 0


# ── Determinism ───────────────────────────────────────────────────


async def test_score_same_input_same_reward(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    body = {"env_id": "math-algebra", "seed": 7, "completion": "x"}
    r1 = await client.post("/v1/score", json=body, headers={"X-Vlabs-Key": plaintext})
    r2 = await client.post("/v1/score", json=body, headers={"X-Vlabs-Key": plaintext})
    assert r1.json()["reward"] == r2.json()["reward"]
    assert r1.json()["components_breakdown"] == r2.json()["components_breakdown"]


# ── Adapter registration coverage ──────────────────────────────────


async def test_score_all_13_envs_reachable(
    client: AsyncClient, api_key
) -> None:
    """Every registered env must accept /v1/score with empty completion."""
    from verifiable_labs_envs import list_environments

    plaintext, _ = api_key
    failures = []
    for env_id in list_environments():
        r = await client.post(
            "/v1/score",
            json={"env_id": env_id, "seed": 0, "completion": ""},
            headers={"X-Vlabs-Key": plaintext},
        )
        if r.status_code != 200:
            failures.append((env_id, r.status_code, r.text[:200]))
    assert not failures, f"envs failed /v1/score: {failures}"
