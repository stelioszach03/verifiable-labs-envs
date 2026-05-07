"""Tests for ``POST /v1/instance`` (Phase 22.B).

Coverage targets:
- happy paths across env families (math, sparse-fourier, mri-knee)
- auth + rate limit reuse from /v1/calibrate
- quota enforcement on scores_per_month
- determinism: same (env_id, seed) returns the same prompt
- env_version pinning
- difficulty_kwargs passthrough
- error paths (unknown env, bad seed, malformed body)
"""
from __future__ import annotations

from datetime import date

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import UsageCounter

# ── Happy paths ────────────────────────────────────────────────────


async def test_instance_math_algebra_returns_prompt(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["instance_seed"] == 0
    assert isinstance(body["prompt"], str) and len(body["prompt"]) > 0
    assert "PROBLEM" in body["prompt"]
    assert isinstance(body["metadata"], dict)
    assert "gold_expr" not in body["metadata"]  # oracle leak check
    assert isinstance(body["env_version"], str)


async def test_instance_math_algebra_multiturn(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra-multiturn", "seed": 1},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["instance_seed"] == 1
    assert "PROBLEM" in body["prompt"]


async def test_instance_math_algebra_tools(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra-tools", "seed": 2},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text


async def test_instance_sparse_fourier(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "sparse-fourier-recovery", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["instance_seed"] == 0
    assert isinstance(body["prompt"], str)
    # numpy array fields in metadata must be coerced to lists
    if "y" in body["metadata"]:
        assert isinstance(body["metadata"]["y"], list)


# ── Determinism ────────────────────────────────────────────────────


async def test_instance_same_seed_returns_same_prompt(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    a = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 42},
        headers={"X-Vlabs-Key": plaintext},
    )
    b = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 42},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json()["prompt"] == b.json()["prompt"]


async def test_instance_different_seeds_differ(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    a = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 1},
        headers={"X-Vlabs-Key": plaintext},
    )
    b = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 99},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert a.json()["prompt"] != b.json()["prompt"]


# ── env_version pinning ────────────────────────────────────────────


async def test_instance_returns_env_version_string(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    from verifiable_labs_envs import __version__

    assert body["env_version"] == __version__


# ── Auth + rate limit reuse ────────────────────────────────────────


async def test_instance_missing_api_key_rejected(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
    )
    assert r.status_code == 401
    assert r.json()["code"] == "invalid_api_key"


async def test_instance_revoked_key_rejected(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    from datetime import UTC, datetime

    from vlabs_api.db import APIKey

    res = await session.execute(select(APIKey).where(APIKey.id == info["api_key_id"]))
    row = res.scalar_one()
    row.revoked_at = datetime.now(UTC)
    await session.commit()

    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 401


async def test_instance_malformed_key_rejected(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": "not-a-valid-key"},
    )
    assert r.status_code == 401


# ── Quota enforcement ──────────────────────────────────────────────


async def test_instance_increments_scores_counter(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200

    res = await session.execute(
        select(UsageCounter).where(
            UsageCounter.api_key_id == info["api_key_id"],
            UsageCounter.month == date.today().replace(day=1),
        )
    )
    counter = res.scalar_one()
    assert counter.scores_count == 1
    # Existing counters (traces/calibrations/etc.) must be 0.
    assert counter.traces_count == 0
    assert counter.calibrations_count == 0


async def test_instance_quota_exhausted_returns_402(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Pre-fill scores_count to the cap; next call must 402."""
    plaintext, info = api_key
    counter = UsageCounter(
        api_key_id=info["api_key_id"],
        month=date.today().replace(day=1),
        scores_count=1_000,  # free-tier cap
    )
    session.add(counter)
    await session.commit()

    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 402
    assert r.json()["code"] == "quota_exceeded"


# ── difficulty_kwargs passthrough ──────────────────────────────────


async def test_instance_difficulty_kwargs_default_empty(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200


async def test_instance_difficulty_kwargs_passed_through(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={
            "env_id": "math-algebra",
            "seed": 0,
            "difficulty_kwargs": {"coef_range": 5},
        },
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text


# ── Error paths ────────────────────────────────────────────────────


async def test_instance_unknown_env_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "does-not-exist", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "unknown_environment"


async def test_instance_negative_seed_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": -1},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_instance_missing_env_id_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_instance_missing_seed_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra"},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_instance_extra_keys_rejected(
    client: AsyncClient, api_key
) -> None:
    """The Pydantic model has ``extra='forbid'`` — clients shouldn't
    smuggle unrecognised top-level fields."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0, "future_field": True},
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


# ── Pre-condition: 13 envs reachable via the registry ──────────────


async def test_instance_all_13_envs_resolvable(
    client: AsyncClient, api_key
) -> None:
    """Each registered env must accept seed=0 via /v1/instance."""
    from verifiable_labs_envs import list_environments

    plaintext, _ = api_key
    failures = []
    for env_id in list_environments():
        r = await client.post(
            "/v1/instance",
            json={"env_id": env_id, "seed": 0},
            headers={"X-Vlabs-Key": plaintext},
        )
        if r.status_code != 200:
            failures.append((env_id, r.status_code, r.text[:200]))
    assert not failures, f"envs failed /v1/instance: {failures}"


# ── 1 KB-ish prompt sanity ──────────────────────────────────────────


async def test_instance_prompt_is_reasonable_length(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    # Symbolic envs produce short prompts (~200 chars). 100 KB ceiling
    # catches accidental array dumps; 1 char floor catches silent failure.
    assert 1 <= len(body["prompt"]) < 100_000


# ── Env metadata excludes oracle fields ────────────────────────────


async def test_instance_metadata_no_gold_expr_for_math(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "math-algebra", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    assert "gold_expr" not in body["metadata"]


async def test_instance_metadata_no_x_true_for_inverse_problem(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/instance",
        json={"env_id": "sparse-fourier-recovery", "seed": 0},
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    assert "x_true" not in body["metadata"]
