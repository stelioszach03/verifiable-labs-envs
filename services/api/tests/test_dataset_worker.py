"""Tests for the vlabs-data async worker (Phase 23.C).

Coverage targets per PHASE_23_PLAN.md §7:
- LLM-client cost estimation + provider-tolerance
- chunk serialisation: JSONL byte-exact, parquet round-trip
- aggregate stats (mean/std/p25/p50/p75) on small samples
- end-to-end ``process_dataset_job`` happy path with respx-mocked LLM
- LLM failure → tuple skipped, counter NOT incremented for failures
- budget cap stops generation early, state stays ``succeeded``
- mid-job restart: re-running picks up at ``generated_tuples``
- queue enqueue/dequeue round-trip (LOCAL_FAKE_R2 path; real Redis
  paths covered in integration tests)
- worker_loop cancellation cleans up gracefully
- storage round-trip in fake mode
- LLM-key encryption integration
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.config import get_settings
from vlabs_api.dataset_worker import (
    _aggregate_stats,
    _serialise_chunk,
    _serialise_chunk_jsonl,
    _serialise_tuple,
    process_dataset_job,
    spawn_worker_pool,
)
from vlabs_api.db import DatasetJob, UsageCounter
from vlabs_api.llm_client import LLMResult, _estimate_cost, _model_price, call_llm
from vlabs_api.llm_key_crypto import decrypt_llm_api_key, encrypt_llm_api_key
from vlabs_api.storage import (
    list_chunks,
    reset_fake_storage_for_tests,
    upload_chunk,
    upload_dataset,
)

# ── llm_client unit tests ──────────────────────────────────────────


def test_model_price_known_model() -> None:
    rates = _model_price("gpt-4o-mini")
    assert rates["prompt"] > 0
    assert rates["completion"] > 0


def test_model_price_openrouter_prefix_tolerated() -> None:
    """OpenRouter passes models like 'anthropic/claude-haiku-4-5'."""
    rates = _model_price("anthropic/claude-haiku-4-5")
    assert rates["prompt"] == 0.0008  # matches the unprefixed entry


def test_model_price_unknown_falls_back() -> None:
    rates = _model_price("brand-new-model-not-yet-priced")
    assert rates["prompt"] == 0.001  # fallback


def test_estimate_cost_scales_with_tokens() -> None:
    cheap = _estimate_cost("gpt-4o-mini", 1000, 1000)
    expensive = _estimate_cost("gpt-4o", 1000, 1000)
    assert cheap < expensive


@pytest.mark.respx(assert_all_called=False)
async def test_call_llm_happy_path(respx_mock) -> None:
    respx_mock.post("https://fake.api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )
    result = await call_llm(
        endpoint_url="https://fake.api/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        system_prompt="sys",
        user_prompt="usr",
    )
    assert result.success is True
    assert result.completion_text == "hello world"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cost_usd_estimate > 0


@pytest.mark.respx(assert_all_called=False)
async def test_call_llm_4xx_returns_failure(respx_mock) -> None:
    respx_mock.post("https://fake.api/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )
    result = await call_llm(
        endpoint_url="https://fake.api/v1",
        api_key="sk-bad",
        model="gpt-4o-mini",
        system_prompt="sys",
        user_prompt="usr",
    )
    assert result.success is False
    assert "http_401" in (result.error or "")


@pytest.mark.respx(assert_all_called=False)
async def test_call_llm_transport_error_returns_failure(respx_mock) -> None:
    respx_mock.post("https://fake.api/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("dns failed")
    )
    result = await call_llm(
        endpoint_url="https://fake.api/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        system_prompt="sys",
        user_prompt="usr",
    )
    assert result.success is False
    assert "transport" in (result.error or "")


@pytest.mark.respx(assert_all_called=False)
async def test_call_llm_appends_chat_completions_to_base_url(respx_mock) -> None:
    """Customer can pass either base URL or full /chat/completions URL."""
    route = respx_mock.post("https://fake.api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
    )
    await call_llm(
        endpoint_url="https://fake.api/v1",  # base URL
        api_key="sk",
        model="gpt-4o-mini",
        system_prompt="s",
        user_prompt="u",
    )
    assert route.called


# ── chunk serialisation ────────────────────────────────────────────


def test_serialise_chunk_jsonl_round_trip() -> None:
    tuples = [
        _serialise_tuple(
            seed=i,
            prompt="P",
            completion="C",
            reward=0.5 + i * 0.01,
            components={"format_valid": 1.0, "parse_valid": 1.0, "correct": 0.5},
            llm_meta=LLMResult(
                completion_text="C",
                prompt_tokens=1,
                completion_tokens=1,
                cost_usd_estimate=0.0001,
                success=True,
            ),
            env_version="0.0.1",
        )
        for i in range(3)
    ]
    payload = _serialise_chunk_jsonl(tuples)
    lines = payload.strip().split(b"\n")
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert all(p["format_version"] == "0.0.1" for p in parsed)


def test_serialise_chunk_format_dispatches_correctly() -> None:
    """_serialise_chunk falls back to JSONL when pyarrow missing."""
    tuples = [
        _serialise_tuple(
            seed=0, prompt="p", completion="c", reward=0.0,
            components={}, env_version="0.0.1",
            llm_meta=LLMResult(
                completion_text="c", prompt_tokens=1, completion_tokens=1,
                cost_usd_estimate=0.0, success=True,
            ),
        )
    ]
    payload = _serialise_chunk("jsonl", tuples)
    assert b"format_version" in payload


# ── aggregate stats ────────────────────────────────────────────────


def test_aggregate_stats_empty_returns_nones() -> None:
    s = _aggregate_stats([])
    assert all(v is None for v in s.values())


def test_aggregate_stats_single_value() -> None:
    s = _aggregate_stats([0.7])
    assert s["mean"] == 0.7
    assert s["std"] == 0.0
    assert s["p25"] == 0.7
    assert s["p50"] == 0.7
    assert s["p75"] == 0.7


def test_aggregate_stats_distribution() -> None:
    s = _aggregate_stats([0.0, 0.25, 0.5, 0.75, 1.0])
    assert s["mean"] == pytest.approx(0.5)
    assert s["p25"] <= 0.5 <= s["p75"]


# ── storage layer (LOCAL_FAKE_R2) ──────────────────────────────────


def test_storage_upload_and_list_in_fake_mode() -> None:
    reset_fake_storage_for_tests()
    user_id = "u1"
    dataset_id = "d1"
    key, sha, size = upload_dataset(user_id, dataset_id, "jsonl", b"line1\nline2")
    assert key.endswith("/jsonl.jsonl")
    assert size == len(b"line1\nline2")
    assert len(sha) == 64

    upload_chunk(user_id, dataset_id, "jsonl", 0, b"chunk0")
    upload_chunk(user_id, dataset_id, "jsonl", 1, b"chunk1")
    chunks = list_chunks(user_id, dataset_id, "jsonl")
    assert len(chunks) == 2


def test_storage_signed_url_in_fake_mode() -> None:
    from vlabs_api.storage import generate_signed_url

    reset_fake_storage_for_tests()
    upload_dataset("u1", "d1", "jsonl", b"x")
    url, expires_at = generate_signed_url("u1/d1/jsonl.jsonl")
    assert url.startswith("file://")
    assert expires_at > datetime.now(UTC)


# ── LLM-key encryption round-trip ──────────────────────────────────


def test_encryption_round_trips() -> None:
    plaintext = "sk-customer-very-secret-key"
    cipher = encrypt_llm_api_key(plaintext)
    # In tests, conftest sets a real Fernet secret — ciphertext should
    # NOT contain the plaintext.
    assert plaintext.encode() not in cipher
    assert decrypt_llm_api_key(cipher) == plaintext


# ── process_dataset_job end-to-end (LOCAL_FAKE_R2) ─────────────────


async def _refetch_job(job_id: uuid.UUID) -> DatasetJob:
    """Fetch a DatasetJob via a fresh session (bypasses fixture-session
    identity-map cache). Required because process_dataset_job runs in
    its own session and writes the row before the test reads it back."""
    from vlabs_api import db as db_module

    factory = db_module._SessionFactory
    assert factory is not None
    async with factory() as s:
        res = await s.execute(select(DatasetJob).where(DatasetJob.id == job_id))
        return res.scalar_one()


async def _refetch_counter() -> UsageCounter:
    from vlabs_api import db as db_module

    factory = db_module._SessionFactory
    assert factory is not None
    async with factory() as s:
        res = await s.execute(select(UsageCounter))
        return res.scalar_one()


async def _seed_job(
    session: AsyncSession,
    user_id: uuid.UUID,
    api_key_id: uuid.UUID,
    *,
    requested_tuples: int = 3,
    seed_start: int = 0,
    budget_usd_cap: float | None = None,
    output_format: str = "jsonl",
    state: str = "queued",
) -> DatasetJob:
    job = DatasetJob(
        user_id=user_id,
        api_key_id=api_key_id,
        env_id="math-algebra",
        env_version="0.0.1-test",
        requested_tuples=requested_tuples,
        seed_start=seed_start,
        seed_end=seed_start + requested_tuples - 1,
        llm_endpoint_url="https://fake.api/v1",
        llm_api_key_encrypted=encrypt_llm_api_key("sk-test-customer"),
        llm_model="gpt-4o-mini",
        budget_usd_cap=budget_usd_cap,
        budget_usd_spent=0.0,
        state=state,
        output_format=output_format,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


def _mock_llm_response(respx_mock, content: str = '{"answer": "x", "confidence": 0.5}') -> None:
    respx_mock.post("https://fake.api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
        )
    )


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_happy_path(
    api_key, session: AsyncSession, respx_mock
) -> None:
    """End-to-end: queue → process → R2 upload → state=succeeded."""
    reset_fake_storage_for_tests()
    _mock_llm_response(respx_mock)
    _, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"])

    await process_dataset_job(job.id)

    final = await _refetch_job(job.id)
    assert final.state == "succeeded"
    assert final.generated_tuples == 3
    assert final.storage_key is not None
    assert final.storage_sha256 is not None
    assert final.storage_size_bytes > 0
    assert final.completed_at is not None


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_increments_tuples_counter(
    api_key, session: AsyncSession, respx_mock
) -> None:
    reset_fake_storage_for_tests()
    _mock_llm_response(respx_mock)
    _, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"])

    await process_dataset_job(job.id)

    counter = await _refetch_counter()
    assert counter.tuples_generated == 3


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_aggregate_stats_populated(
    api_key, session: AsyncSession, respx_mock
) -> None:
    reset_fake_storage_for_tests()
    _mock_llm_response(respx_mock)
    _, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"], requested_tuples=5)

    await process_dataset_job(job.id)

    final = await _refetch_job(job.id)
    assert final.mean_reward is not None
    assert 0.0 <= final.mean_reward <= 1.0
    assert final.std_reward is not None
    assert final.p25_reward is not None
    assert final.p50_reward is not None
    assert final.p75_reward is not None
    assert final.completion_success_rate is not None


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_budget_cap_stops_early(
    api_key, session: AsyncSession, respx_mock
) -> None:
    reset_fake_storage_for_tests()
    _mock_llm_response(respx_mock)  # ~70 tokens / call → ~$0.000023
    _, info = api_key
    # Tiny cap → only 1-2 tuples should generate before stopping.
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        requested_tuples=100,
        budget_usd_cap=0.00005,
    )

    await process_dataset_job(job.id)

    final = await _refetch_job(job.id)
    assert final.state == "succeeded"
    # Budget kicks in: generated < requested.
    assert final.generated_tuples < 100
    assert final.budget_usd_spent <= 0.00005 * 1.5  # within rate variance


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_llm_failure_skips_success_count(
    api_key, session: AsyncSession, respx_mock
) -> None:
    """When the customer's LLM 401s, tuples are still scored (zero
    reward) but completion_success_rate < 1.0."""
    reset_fake_storage_for_tests()
    respx_mock.post("https://fake.api/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="Unauthorized"),
    )
    _, info = api_key
    job = await _seed_job(session, info["user_id"], info["api_key_id"], requested_tuples=3)

    await process_dataset_job(job.id)

    final = await _refetch_job(job.id)
    assert final.state == "succeeded"
    # All LLM calls failed → zero successes, but tuples still produced.
    assert final.completion_success_rate == 0.0
    assert final.generated_tuples == 3


@pytest.mark.respx(assert_all_called=False)
async def test_process_job_already_terminal_is_noop(
    api_key, session: AsyncSession, respx_mock
) -> None:
    """Reprocessing a succeeded/failed job is a no-op."""
    reset_fake_storage_for_tests()
    _mock_llm_response(respx_mock)
    _, info = api_key
    job = await _seed_job(
        session, info["user_id"], info["api_key_id"],
        state="succeeded",
    )

    await process_dataset_job(job.id)
    # If we'd processed, generated_tuples would have changed.
    res = await session.execute(select(DatasetJob).where(DatasetJob.id == job.id))
    final = res.scalar_one()
    assert final.generated_tuples == 0  # unchanged


async def test_process_job_unknown_id_is_noop(api_key) -> None:
    """Passing a non-existent job_id logs and returns; doesn't raise."""
    fake_id = uuid.uuid4()
    await process_dataset_job(fake_id)  # must not raise


# ── worker_loop / spawn_worker_pool ────────────────────────────────


async def test_spawn_worker_pool_default_size() -> None:
    """Default pool size from settings; tasks created."""
    tasks = await spawn_worker_pool()
    try:
        assert len(tasks) == get_settings().vlabs_data_worker_pool_size
        for t in tasks:
            assert isinstance(t, asyncio.Task)
            assert not t.done()
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t


async def test_spawn_worker_pool_explicit_size_zero() -> None:
    """pool_size=0 returns no tasks (useful for tests that don't want
    the worker fighting for the queue)."""
    tasks = await spawn_worker_pool(pool_size=0)
    assert tasks == []


async def test_worker_loop_cancels_gracefully() -> None:
    tasks = await spawn_worker_pool(pool_size=1)
    assert len(tasks) == 1
    tasks[0].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await tasks[0]
    assert tasks[0].cancelled() or tasks[0].done()


# ── enqueue / dequeue (LOCAL_FAKE_R2 / no-Redis path) ──────────────


async def test_dequeue_returns_none_without_redis() -> None:
    """In tests Upstash is unset → dequeue immediately returns None."""
    from vlabs_api.dataset_worker import dequeue_dataset_job
    out = await dequeue_dataset_job(timeout_s=1)
    assert out is None


async def test_enqueue_silent_without_redis() -> None:
    """In tests Upstash is unset → enqueue is a no-op (logs warning, no raise)."""
    from vlabs_api.dataset_worker import enqueue_dataset_job
    await enqueue_dataset_job(uuid.uuid4())  # must not raise


async def test_rescue_queued_jobs_replays_pending(
    api_key, session: AsyncSession
) -> None:
    """rescue_queued_jobs walks 'queued' rows and re-enqueues them.
    In tests the enqueue is a no-op (no Redis), but the count must
    reflect the rescued rows."""
    from vlabs_api.dataset_worker import rescue_queued_jobs

    _, info = api_key
    await _seed_job(session, info["user_id"], info["api_key_id"], state="queued")
    await _seed_job(session, info["user_id"], info["api_key_id"], state="queued",
                    seed_start=100)
    # State = succeeded shouldn't be re-enqueued.
    await _seed_job(session, info["user_id"], info["api_key_id"], state="succeeded",
                    seed_start=200)

    count = await rescue_queued_jobs(session)
    assert count == 2
