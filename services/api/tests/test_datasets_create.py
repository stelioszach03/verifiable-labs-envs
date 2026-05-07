"""Tests for ``POST /v1/datasets`` (Phase 23.B).

Coverage targets per PHASE_23_PLAN.md:
- happy-path job creation (queued state, dataset_id format, env_version pin)
- LLM API key encryption: stored as ciphertext, decryptable to plaintext
- LLM API key NEVER returned in response
- quota enforcement on tuples_per_month (pre-flight check)
- idempotency (D6 ruling): in-window cache, out-of-window delete-and-replace
- auth: missing key → 401; revoked key → 401
- unknown env_id → 404
- Pydantic validation: requested_tuples bounds, seed_start, llm_endpoint_url, …
- output_format defaults to "parquet"; "jsonl" accepted; other rejected
- budget_usd_cap optional; positive only
- seed_end derived correctly (seed_start + requested_tuples - 1)
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import APIKey, DatasetJob, UsageCounter, User
from vlabs_api.llm_key_crypto import decrypt_llm_api_key


def _good_payload(**overrides) -> dict:
    """Minimal valid POST /v1/datasets body."""
    base = {
        "env_id": "math-algebra",
        "requested_tuples": 10,
        "seed_start": 0,
        "llm_endpoint_url": "https://api.openai.com/v1",
        "llm_api_key": "sk-test-customer-key",
        "llm_model": "gpt-4o-mini",
    }
    base.update(overrides)
    return base


# ── happy paths ───────────────────────────────────────────────────


async def test_create_dataset_returns_201_with_queued_state(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["dataset_id"].startswith("ds_")
    assert body["state"] == "queued"
    assert body["requested_tuples"] == 10
    assert body["seed_start"] == 0
    assert body["seed_end"] == 9  # seed_start + requested_tuples - 1
    assert body["output_format"] == "parquet"


async def test_create_dataset_persists_row(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=50, seed_start=100),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201

    res = await session.execute(select(DatasetJob))
    row = res.scalar_one()
    assert row.user_id == info["user_id"]
    assert row.api_key_id == info["api_key_id"]
    assert row.env_id == "math-algebra"
    assert row.requested_tuples == 50
    assert row.seed_start == 100
    assert row.seed_end == 149
    assert row.state == "queued"
    assert row.generated_tuples == 0


async def test_create_dataset_pins_env_version(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    from verifiable_labs_envs import __version__

    assert r.json()["env_version"] == __version__
    res = await session.execute(select(DatasetJob))
    row = res.scalar_one()
    assert row.env_version == __version__


async def test_create_dataset_supports_jsonl_output_format(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(output_format="jsonl"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201
    assert r.json()["output_format"] == "jsonl"


async def test_create_dataset_optional_budget_cap(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(budget_usd_cap=1.50),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201
    res = await session.execute(select(DatasetJob))
    row = res.scalar_one()
    assert row.budget_usd_cap == pytest.approx(1.50)


# ── LLM API-key encryption (D1-B) ─────────────────────────────────


async def test_create_dataset_encrypts_llm_api_key(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    secret = "sk-customer-secret-not-for-anyone"
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(llm_api_key=secret),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201
    res = await session.execute(select(DatasetJob))
    row = res.scalar_one()
    # Encrypted bytes must NOT contain the plaintext key.
    assert secret.encode() not in row.llm_api_key_encrypted
    # But decryption round-trips.
    assert decrypt_llm_api_key(row.llm_api_key_encrypted) == secret


async def test_create_dataset_response_never_returns_llm_api_key(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    secret = "sk-very-secret-customer-key"
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(llm_api_key=secret),
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    # No field exposes the LLM key.
    assert "llm_api_key" not in body
    assert "api_key" not in body
    assert secret not in r.text


# ── Quota enforcement (D8) ────────────────────────────────────────


async def test_create_dataset_blocks_when_quota_exhausted(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    from datetime import date

    counter = UsageCounter(
        api_key_id=info["api_key_id"],
        month=date.today().replace(day=1),
        tuples_generated=995,  # 5 left on the free 1000 cap
    )
    session.add(counter)
    await session.commit()

    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=10),  # 995 + 10 > 1000
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 402
    assert r.json()["code"] == "quota_exceeded"


async def test_create_dataset_does_not_increment_counter_yet(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """The counter increments per-tuple in 23.C as the worker generates;
    creation alone does NOT debit."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=100),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 201
    res = await session.execute(select(UsageCounter))
    counter = res.scalar_one_or_none()
    if counter is not None:
        assert counter.tuples_generated == 0


# ── Idempotency (D6) ──────────────────────────────────────────────


async def test_idempotent_create_returns_cached_job(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="test-1"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.status_code == 201
    dataset_id_1 = r1.json()["dataset_id"]

    # Second call with the SAME key + DIFFERENT body returns the cached row.
    r2 = await client.post(
        "/v1/datasets",
        json=_good_payload(
            idempotency_key="test-1",
            requested_tuples=999,  # different; should be ignored
        ),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r2.status_code == 201
    assert r2.json()["dataset_id"] == dataset_id_1
    # Original requested_tuples preserved (not overwritten).
    assert r2.json()["requested_tuples"] == 10

    # Only one row in the DB.
    res = await session.execute(select(DatasetJob))
    rows = res.scalars().all()
    assert len(rows) == 1


async def test_idempotent_different_keys_distinct_jobs(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="key-A"),
        headers={"X-Vlabs-Key": plaintext},
    )
    r2 = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="key-B"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.json()["dataset_id"] != r2.json()["dataset_id"]


async def test_idempotent_no_key_distinct_jobs(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    r2 = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.json()["dataset_id"] != r2.json()["dataset_id"]


async def test_idempotent_window_24h_stale_replaces(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, _ = api_key
    r1 = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="stale"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r1.status_code == 201

    # Back-date the row beyond the window.
    res = await session.execute(select(DatasetJob))
    row = res.scalar_one()
    row.created_at = datetime.now(UTC) - timedelta(hours=25)
    await session.commit()

    # Re-issue same key — should NOT match (out of window).
    r2 = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="stale"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r2.status_code == 201
    assert r2.json()["dataset_id"] != r1.json()["dataset_id"]


# ── Auth ──────────────────────────────────────────────────────────


async def test_create_dataset_missing_api_key_rejected(
    client: AsyncClient
) -> None:
    r = await client.post("/v1/datasets", json=_good_payload())
    assert r.status_code == 401


async def test_create_dataset_revoked_key_rejected(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    res = await session.execute(select(APIKey).where(APIKey.id == info["api_key_id"]))
    row = res.scalar_one()
    row.revoked_at = datetime.now(UTC)
    await session.commit()

    r = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 401


# ── Error paths ────────────────────────────────────────────────────


async def test_create_dataset_unknown_env_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(env_id="not-a-real-env"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "unknown_environment"


async def test_create_dataset_zero_tuples_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=0),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_excess_tuples_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=100_001),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_negative_seed_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(seed_start=-1),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_invalid_output_format_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(output_format="csv"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_negative_budget_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(budget_usd_cap=-0.5),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_zero_budget_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(budget_usd_cap=0),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_extra_keys_rejected(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    body = _good_payload()
    body["future_field"] = True
    r = await client.post(
        "/v1/datasets",
        json=body,
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


async def test_create_dataset_idempotency_key_too_long_returns_422(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="k" * 201),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


# ── Cross-user isolation ──────────────────────────────────────────


async def test_create_dataset_isolated_per_user(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Two users with the same idempotency_key get distinct jobs."""
    plaintext_a, _ = api_key

    # Create user B + key.
    user_b = User(email="b@example.com")
    session.add(user_b)
    await session.flush()
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="other-key",
    )
    session.add(key_b)
    await session.commit()

    r_a = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="shared"),
        headers={"X-Vlabs-Key": plaintext_a},
    )
    r_b = await client.post(
        "/v1/datasets",
        json=_good_payload(idempotency_key="shared"),
        headers={"X-Vlabs-Key": plaintext_b},
    )
    assert r_a.status_code == 201
    assert r_b.status_code == 201
    assert r_a.json()["dataset_id"] != r_b.json()["dataset_id"]
