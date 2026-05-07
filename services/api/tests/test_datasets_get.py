"""Tests for the read-side dataset endpoints (Phase 23.D).

Coverage targets per PHASE_23_PLAN.md §5.D6 + §11:
- ``GET /v1/datasets/{id}`` — single status (happy + 404 paths,
  cross-user isolation, full payload incl. aggregate stats, no LLM
  API key leak)
- ``GET /v1/datasets`` — paginated list (default/custom limit + offset,
  state filter, ordering by created_at desc, cross-user isolation)
- ``GET /v1/datasets/{id}/download`` — 302 redirect default,
  ``Accept: application/json`` returns inline metadata; 409 for
  non-succeeded states; 404 for unknown / cross-user
- auth: missing key → 401 on every endpoint; malformed dataset_id → 404
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import APIKey, DatasetJob, User
from vlabs_api.llm_key_crypto import encrypt_llm_api_key


async def _seed_job(
    session: AsyncSession,
    user_id: uuid.UUID,
    api_key_id: uuid.UUID,
    *,
    state: str = "queued",
    requested_tuples: int = 10,
    seed_start: int = 0,
    output_format: str = "jsonl",
    storage_key: str | None = None,
    storage_sha256: str | None = None,
    storage_size_bytes: int | None = None,
    mean_reward: float | None = None,
    completion_success_rate: float | None = None,
    generated_tuples: int = 0,
    budget_usd_spent: float = 0.0,
    error: str | None = None,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> DatasetJob:
    job = DatasetJob(
        user_id=user_id,
        api_key_id=api_key_id,
        env_id="math-algebra",
        env_version="0.0.1-test",
        requested_tuples=requested_tuples,
        seed_start=seed_start,
        seed_end=seed_start + requested_tuples - 1,
        llm_endpoint_url="https://api.openai.com/v1",
        llm_api_key_encrypted=encrypt_llm_api_key("sk-test-customer-key"),
        llm_model="gpt-4o-mini",
        output_format=output_format,
        state=state,
        generated_tuples=generated_tuples,
        budget_usd_spent=budget_usd_spent,
        storage_key=storage_key,
        storage_sha256=storage_sha256,
        storage_size_bytes=storage_size_bytes,
        mean_reward=mean_reward,
        completion_success_rate=completion_success_rate,
        error=error,
        idempotency_key=idempotency_key,
    )
    if completed_at is not None:
        job.completed_at = completed_at
    session.add(job)
    await session.flush()
    if created_at is not None:
        job.created_at = created_at
    await session.commit()
    await session.refresh(job)
    return job


# ── GET /v1/datasets/{dataset_id} ─────────────────────────────────


async def test_get_dataset_happy_path(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"])
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset_id"] == f"ds_{job.id.hex}"
    assert body["env_id"] == "math-algebra"
    assert body["state"] == "queued"
    assert body["requested_tuples"] == 10
    assert body["generated_tuples"] == 0


async def test_get_dataset_does_not_return_llm_api_key(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """The encrypted-key field MUST NOT appear in the response payload."""
    plaintext, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"])
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    body = r.json()
    # Neither the plaintext nor any encrypted-key field should leak.
    assert "llm_api_key" not in body
    assert "llm_api_key_encrypted" not in body
    assert "sk-test-customer-key" not in r.text
    # But endpoint URL + model are part of the public contract.
    assert body["llm_endpoint_url"] == "https://api.openai.com/v1"
    assert body["llm_model"] == "gpt-4o-mini"


async def test_get_dataset_returns_aggregate_stats_when_succeeded(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session,
        info["user_id"],
        info["api_key_id"],
        state="succeeded",
        generated_tuples=10,
        storage_key=f"{info['user_id']}/dataset/jsonl.jsonl",
        storage_sha256="a" * 64,
        storage_size_bytes=1024,
        mean_reward=0.42,
        completion_success_rate=0.9,
        completed_at=datetime.now(UTC),
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "succeeded"
    assert body["mean_reward"] == 0.42
    assert body["completion_success_rate"] == 0.9
    assert body["storage_sha256"] == "a" * 64
    assert body["storage_size_bytes"] == 1024
    assert body["completed_at"] is not None


async def test_get_dataset_unknown_id_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    fake = f"ds_{uuid.uuid4().hex}"
    r = await client.get(
        f"/v1/datasets/{fake}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "dataset_job_not_found"


async def test_get_dataset_malformed_id_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/datasets/not-a-real-id",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "dataset_job_not_found"


async def test_get_dataset_cross_user_returns_404(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """User B cannot read user A's job — same 404 surface as missing.

    This protects against information leak (if we 403'd we'd leak the
    fact that the id exists)."""
    _, info_a = api_key
    job = await _seed_job(session, info_a["user_id"], info_a["api_key_id"])

    # Build a second user + key.
    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    user_b = User(email=f"u-{uuid.uuid4().hex[:8]}@example.com", name="B")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="B-key",
    )
    session.add(key_b)
    await session.commit()

    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}",
        headers={"X-Vlabs-Key": plaintext_b},
    )
    assert r.status_code == 404


async def test_get_dataset_no_auth_returns_401(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    _, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"])
    r = await client.get(f"/v1/datasets/ds_{job.id.hex}")
    assert r.status_code == 401


# ── GET /v1/datasets ──────────────────────────────────────────────


async def test_list_datasets_empty(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/datasets", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 100
    assert body["offset"] == 0


async def test_list_datasets_returns_user_jobs(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    await _seed_job(session, info["user_id"], info["api_key_id"], seed_start=0)
    await _seed_job(session, info["user_id"], info["api_key_id"], seed_start=100)

    r = await client.get(
        "/v1/datasets", headers={"X-Vlabs-Key": plaintext}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert item["dataset_id"].startswith("ds_")
        assert item["env_id"] == "math-algebra"


async def test_list_datasets_sorted_by_created_at_desc(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    older = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        seed_start=0,
        created_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        seed_start=100,
        created_at=datetime.now(UTC),
    )

    r = await client.get(
        "/v1/datasets", headers={"X-Vlabs-Key": plaintext}
    )
    items = r.json()["items"]
    assert items[0]["dataset_id"] == f"ds_{newer.id.hex}"
    assert items[1]["dataset_id"] == f"ds_{older.id.hex}"


async def test_list_datasets_pagination_limit_offset(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    for i in range(5):
        await _seed_job(
            session, info["user_id"], info["api_key_id"], seed_start=i * 10
        )

    r = await client.get(
        "/v1/datasets?limit=2&offset=1",
        headers={"X-Vlabs-Key": plaintext},
    )
    body = r.json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert len(body["items"]) == 2


async def test_list_datasets_state_filter(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    await _seed_job(
        session, info["user_id"], info["api_key_id"],
        seed_start=0, state="queued",
    )
    await _seed_job(
        session, info["user_id"], info["api_key_id"],
        seed_start=100, state="succeeded",
    )
    await _seed_job(
        session, info["user_id"], info["api_key_id"],
        seed_start=200, state="failed",
    )

    r = await client.get(
        "/v1/datasets?state=succeeded",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["state"] == "succeeded"


async def test_list_datasets_cross_user_isolation(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """User B's list excludes user A's jobs."""
    _, info_a = api_key
    await _seed_job(session, info_a["user_id"], info_a["api_key_id"])

    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    user_b = User(email=f"u-{uuid.uuid4().hex[:8]}@example.com", name="B")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="B-key",
    )
    session.add(key_b)
    await session.commit()

    r = await client.get(
        "/v1/datasets", headers={"X-Vlabs-Key": plaintext_b}
    )
    assert r.json()["total"] == 0


async def test_list_datasets_no_auth_returns_401(client: AsyncClient) -> None:
    r = await client.get("/v1/datasets")
    assert r.status_code == 401


async def test_list_datasets_limit_bounds(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r1 = await client.get(
        "/v1/datasets?limit=0", headers={"X-Vlabs-Key": plaintext}
    )
    r2 = await client.get(
        "/v1/datasets?limit=10000", headers={"X-Vlabs-Key": plaintext}
    )
    assert r1.status_code == 422
    assert r2.status_code == 422


# ── GET /v1/datasets/{id}/download ────────────────────────────────


async def test_download_succeeded_returns_302_redirect_by_default(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="succeeded",
        generated_tuples=10,
        storage_key=f"{info['user_id']}/dataset/jsonl.jsonl",
        storage_sha256="a" * 64,
        storage_size_bytes=42,
        completed_at=datetime.now(UTC),
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext},
        follow_redirects=False,
    )
    assert r.status_code == 302
    # In LOCAL_FAKE_R2 mode the URL is a file:// reference.
    assert r.headers["location"].startswith("file://")


async def test_download_returns_json_when_requested(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="succeeded",
        generated_tuples=10,
        storage_key=f"{info['user_id']}/dataset/jsonl.jsonl",
        storage_sha256="b" * 64,
        storage_size_bytes=4096,
        completed_at=datetime.now(UTC),
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={
            "X-Vlabs-Key": plaintext,
            "Accept": "application/json",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == f"ds_{job.id.hex}"
    assert body["download_url"].startswith("file://")
    assert body["sha256"] == "b" * 64
    assert body["size_bytes"] == 4096
    assert body["output_format"] == "jsonl"
    assert "expires_at" in body


async def test_download_queued_returns_409(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"], state="queued"
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 409
    assert r.json()["code"] == "dataset_job_invalid_state"


async def test_download_running_returns_409(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"], state="running"
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 409


async def test_download_failed_returns_409(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="failed",
        error="LLM transport error",
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 409


async def test_download_succeeded_without_storage_key_returns_409(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Defensive — succeeded but storage_key NULL is invalid state.

    Shouldn't happen in practice (worker writes them in the same
    transaction), but the endpoint guards against it explicitly.
    """
    plaintext, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="succeeded",
        completed_at=datetime.now(UTC),
        # storage_key intentionally None
    )
    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 409


async def test_download_unknown_id_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    fake = f"ds_{uuid.uuid4().hex}"
    r = await client.get(
        f"/v1/datasets/{fake}/download",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404


async def test_download_cross_user_returns_404(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    _, info_a = api_key
    job = await _seed_job(
        session, info_a["user_id"], info_a["api_key_id"],
        state="succeeded",
        storage_key=f"{info_a['user_id']}/d/jsonl.jsonl",
        storage_sha256="c" * 64,
        storage_size_bytes=10,
        completed_at=datetime.now(UTC),
    )

    from vlabs_api.auth import (
        generate_plaintext_key,
        hash_plaintext_key,
        key_prefix,
    )
    user_b = User(email=f"u-{uuid.uuid4().hex[:8]}@example.com", name="B")
    session.add(user_b)
    await session.flush()
    plaintext_b = generate_plaintext_key()
    key_b = APIKey(
        user_id=user_b.id,
        key_hash=hash_plaintext_key(plaintext_b),
        key_prefix=key_prefix(plaintext_b),
        name="B-key",
    )
    session.add(key_b)
    await session.commit()

    r = await client.get(
        f"/v1/datasets/ds_{job.id.hex}/download",
        headers={"X-Vlabs-Key": plaintext_b},
    )
    assert r.status_code == 404


async def test_download_no_auth_returns_401(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    _, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="succeeded",
        storage_key=f"{info['user_id']}/d/jsonl.jsonl",
        storage_sha256="d" * 64,
        storage_size_bytes=1,
        completed_at=datetime.now(UTC),
    )
    r = await client.get(f"/v1/datasets/ds_{job.id.hex}/download")
    assert r.status_code == 401
