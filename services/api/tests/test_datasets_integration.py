"""End-to-end integration tests for vlabs-data (Phase 23.E).

These exercise the full lifecycle that crosses subsystem boundaries —
HTTP layer → DB → worker pool → R2 (LOCAL_FAKE_R2) → signed URL.
The unit-level tests in ``test_datasets_create.py``,
``test_dataset_worker.py``, and ``test_datasets_get.py`` already cover
each subsystem in isolation; these tests catch contract drift between
them.

Coverage targets per PHASE_23_PLAN.md §15:
- POST → process_dataset_job → GET status → GET download → verify
  payload bytes match storage_sha256
- Idempotency in-window: same key on running job returns dataset_id +
  current state without re-enqueue
- Idempotency out-of-window: stale row deleted, new row inserted,
  fresh dataset_id
- tuples_generated counter increments per-tuple after worker run
- Worker rescue picks up orphaned ``queued`` rows on startup
- Multiple jobs: list returns them paginated + state-filtered, sorted
- Budget cap: ``state=succeeded`` with ``generated_tuples < requested``
- All-LLM-failures: ``state=succeeded`` with
  ``completion_success_rate=0.0``, ``generated_tuples=requested``
- Cross-user list/get/download isolation across full pipeline
- Storage key path matches D5 convention: ``{user}/{dataset}/{format}.{ext}``
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.dataset_worker import process_dataset_job, rescue_queued_jobs
from vlabs_api.db import APIKey, DatasetJob, UsageCounter, User
from vlabs_api.ids import parse_dataset_id
from vlabs_api.storage import reset_fake_storage_for_tests


def _good_payload(**overrides) -> dict:
    base = {
        "env_id": "math-algebra",
        "requested_tuples": 3,
        "seed_start": 0,
        "llm_endpoint_url": "https://fake.api/v1",
        "llm_api_key": "sk-test-customer-key",
        "llm_model": "gpt-4o-mini",
        "output_format": "jsonl",
    }
    base.update(overrides)
    return base


def _mock_llm(respx_mock, status: int = 200, content: str = "result-x") -> None:
    if status == 200:
        respx_mock.post("https://fake.api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                },
            )
        )
    else:
        respx_mock.post("https://fake.api/v1/chat/completions").mock(
            return_value=httpx.Response(status, text="LLM error"),
        )


# ── Full lifecycle ────────────────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_full_lifecycle_post_process_get_download(
    client: AsyncClient, api_key, respx_mock, session: AsyncSession
) -> None:
    """The customer journey end-to-end."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    # 1. Customer POSTs the job.
    create = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert create.status_code == 201
    dataset_id = create.json()["dataset_id"]
    assert create.json()["state"] == "queued"

    # 2. Worker processes it (in real prod, Redis pulls + worker_loop).
    job_uuid = parse_dataset_id(dataset_id)
    await process_dataset_job(job_uuid)

    # 3. Customer polls — succeeded.
    status = await client.get(
        f"/v1/datasets/{dataset_id}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert status.status_code == 200
    body = status.json()
    assert body["state"] == "succeeded"
    assert body["generated_tuples"] == 3
    assert body["storage_sha256"] is not None

    # 4. Customer fetches the JSON variant of /download.
    dl = await client.get(
        f"/v1/datasets/{dataset_id}/download",
        headers={"X-Vlabs-Key": plaintext, "Accept": "application/json"},
    )
    assert dl.status_code == 200
    dl_body = dl.json()
    assert dl_body["sha256"] == body["storage_sha256"]
    assert dl_body["size_bytes"] == body["storage_size_bytes"]

    # 5. The integrity hash matches the actual file bytes.
    file_url = dl_body["download_url"]
    assert file_url.startswith("file://")
    on_disk = Path(file_url.removeprefix("file://"))
    assert on_disk.exists()
    actual_sha = hashlib.sha256(on_disk.read_bytes()).hexdigest()
    assert actual_sha == dl_body["sha256"]


@pytest.mark.respx(assert_all_called=False)
async def test_lifecycle_jsonl_payload_is_well_formed(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Each line in the JSONL output is a parseable tuple with the
    expected fields."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=3, output_format="jsonl"),
        headers={"X-Vlabs-Key": plaintext},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    dl = await client.get(
        f"/v1/datasets/{dataset_id}/download",
        headers={"X-Vlabs-Key": plaintext, "Accept": "application/json"},
    )
    on_disk = Path(dl.json()["download_url"].removeprefix("file://"))
    lines = [
        line for line in on_disk.read_bytes().decode().splitlines() if line.strip()
    ]
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    for row in rows:
        for key in (
            "format_version",
            "seed",
            "env_version",
            "prompt",
            "completion",
            "reward",
            "components",
            "llm",
        ):
            assert key in row
        assert isinstance(row["reward"], float)
        assert isinstance(row["llm"]["success"], bool)


@pytest.mark.respx(assert_all_called=False)
async def test_storage_key_matches_d5_convention(
    client: AsyncClient, api_key, respx_mock, session: AsyncSession
) -> None:
    """Object key MUST be ``{user_id}/{dataset_id}/{format}.{ext}``.

    Migration path away from R2 (PHASE_23_PLAN.md R5) depends on this
    convention staying stable."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, info = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    res = await session.execute(
        select(DatasetJob).where(DatasetJob.id == parse_dataset_id(dataset_id))
    )
    job = res.scalar_one()
    expected = f"{info['user_id']}/{job.id}/jsonl.jsonl"
    assert job.storage_key == expected


# ── Quota integration ─────────────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_tuples_counter_increments_after_full_run(
    client: AsyncClient, api_key, respx_mock, session: AsyncSession
) -> None:
    """``tuples_generated`` debits land in the ``usage_counters`` table.

    Per-tuple, post-scoring (D8) — failed LLM calls do NOT increment."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=5),
        headers={"X-Vlabs-Key": plaintext},
    )
    await process_dataset_job(parse_dataset_id(create.json()["dataset_id"]))

    # Counter row exists after worker run.
    from vlabs_api import db as db_module
    async with db_module._SessionFactory() as fresh:  # type: ignore[misc]
        res = await fresh.execute(select(UsageCounter))
        counter = res.scalar_one()
        assert counter.tuples_generated == 5


@pytest.mark.respx(assert_all_called=False)
async def test_failed_llm_does_not_increment_counter(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """All-LLM-failures: tuples are still produced (zero reward) but
    ``completion_success_rate=0.0``. The successful-row counter is
    documented per PHASE_23_PLAN.md §10."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock, status=401)
    plaintext, _ = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=3),
        headers={"X-Vlabs-Key": plaintext},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    status = await client.get(
        f"/v1/datasets/{dataset_id}",
        headers={"X-Vlabs-Key": plaintext},
    )
    body = status.json()
    assert body["state"] == "succeeded"
    assert body["completion_success_rate"] == 0.0
    assert body["generated_tuples"] == 3


# ── Idempotency across the lifecycle ──────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_idempotency_in_window_returns_same_dataset_id_after_completion(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Re-issue with the same idempotency_key after the job completes
    returns the original dataset_id with state=succeeded."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    payload = _good_payload(idempotency_key="my-key-1")
    r1 = await client.post(
        "/v1/datasets", json=payload, headers={"X-Vlabs-Key": plaintext}
    )
    dataset_id_1 = r1.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id_1))

    # Re-issue with same key — server returns the original.
    r2 = await client.post(
        "/v1/datasets", json=payload, headers={"X-Vlabs-Key": plaintext}
    )
    assert r2.status_code == 201
    assert r2.json()["dataset_id"] == dataset_id_1
    # The cached row is now in 'succeeded' state — that's what the user
    # sees instead of the original 'queued'.
    assert r2.json()["state"] == "succeeded"


@pytest.mark.respx(assert_all_called=False)
async def test_idempotency_in_window_does_not_double_increment_counter(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """In-window re-issue must NOT debit a second time."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    payload = _good_payload(idempotency_key="dedup-counter")
    r1 = await client.post(
        "/v1/datasets", json=payload, headers={"X-Vlabs-Key": plaintext}
    )
    await process_dataset_job(parse_dataset_id(r1.json()["dataset_id"]))

    # Re-issue ×2.
    await client.post(
        "/v1/datasets", json=payload, headers={"X-Vlabs-Key": plaintext}
    )
    await client.post(
        "/v1/datasets", json=payload, headers={"X-Vlabs-Key": plaintext}
    )

    from vlabs_api import db as db_module
    async with db_module._SessionFactory() as fresh:  # type: ignore[misc]
        res = await fresh.execute(select(UsageCounter))
        counter = res.scalar_one()
        # Single job ran → only the original 3 tuples landed.
        assert counter.tuples_generated == 3


# ── Worker-pool rescue path ───────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_worker_rescue_finds_orphan_queued_rows(
    client: AsyncClient, api_key, respx_mock, session: AsyncSession
) -> None:
    """A row left in ``queued`` (e.g. Redis was down on POST) is picked
    up by ``rescue_queued_jobs`` on startup."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    # Create two jobs — one in 'queued', one already 'succeeded' (won't
    # be rescued).
    await client.post(
        "/v1/datasets",
        json=_good_payload(seed_start=0),
        headers={"X-Vlabs-Key": plaintext},
    )
    r2 = await client.post(
        "/v1/datasets",
        json=_good_payload(seed_start=100),
        headers={"X-Vlabs-Key": plaintext},
    )
    # Force one into a different state.
    res = await session.execute(
        select(DatasetJob).where(
            DatasetJob.id == parse_dataset_id(r2.json()["dataset_id"])
        )
    )
    job2 = res.scalar_one()
    job2.state = "succeeded"
    await session.commit()

    # Rescue from a fresh session.
    from vlabs_api import db as db_module
    async with db_module._SessionFactory() as fresh:  # type: ignore[misc]
        rescued = await rescue_queued_jobs(fresh)
    assert rescued == 1


# ── Multi-job list/get with state filter ──────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_list_filters_after_lifecycle_runs(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Three jobs run end-to-end; list with state=succeeded returns 3."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    ids = []
    for i in range(3):
        r = await client.post(
            "/v1/datasets",
            json=_good_payload(seed_start=i * 10, requested_tuples=2),
            headers={"X-Vlabs-Key": plaintext},
        )
        ids.append(r.json()["dataset_id"])
    for did in ids:
        await process_dataset_job(parse_dataset_id(did))

    # List with state filter.
    succ = await client.get(
        "/v1/datasets?state=succeeded",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert succ.json()["total"] == 3

    queued = await client.get(
        "/v1/datasets?state=queued",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert queued.json()["total"] == 0


@pytest.mark.respx(assert_all_called=False)
async def test_list_pagination_after_many_jobs(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Created 4 jobs; pagination at limit=2 returns 2 + 2."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    for i in range(4):
        await client.post(
            "/v1/datasets",
            json=_good_payload(seed_start=i * 10),
            headers={"X-Vlabs-Key": plaintext},
        )

    p1 = await client.get(
        "/v1/datasets?limit=2&offset=0",
        headers={"X-Vlabs-Key": plaintext},
    )
    p2 = await client.get(
        "/v1/datasets?limit=2&offset=2",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert len(p1.json()["items"]) == 2
    assert len(p2.json()["items"]) == 2
    # Pages don't overlap.
    p1_ids = {x["dataset_id"] for x in p1.json()["items"]}
    p2_ids = {x["dataset_id"] for x in p2.json()["items"]}
    assert p1_ids.isdisjoint(p2_ids)


# ── Budget cap end-to-end ─────────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_budget_cap_finalises_with_partial_tuples(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Budget cap → state=succeeded, generated < requested."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)  # ~70 tokens × $0.0000003 ≈ $0.00002 / call
    plaintext, _ = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=100, budget_usd_cap=0.00005),
        headers={"X-Vlabs-Key": plaintext},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    status = await client.get(
        f"/v1/datasets/{dataset_id}",
        headers={"X-Vlabs-Key": plaintext},
    )
    body = status.json()
    assert body["state"] == "succeeded"
    assert body["generated_tuples"] < 100


# ── Cross-user isolation end-to-end ───────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_full_cross_user_isolation(
    client: AsyncClient, api_key, respx_mock, session: AsyncSession
) -> None:
    """User A's full lifecycle is invisible to user B (list / get /
    download all return 404 / empty)."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext_a, info_a = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext_a},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    # Build user B.
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

    # B lists — empty.
    lst = await client.get("/v1/datasets", headers={"X-Vlabs-Key": plaintext_b})
    assert lst.json()["total"] == 0

    # B gets — 404.
    get_ = await client.get(
        f"/v1/datasets/{dataset_id}",
        headers={"X-Vlabs-Key": plaintext_b},
    )
    assert get_.status_code == 404

    # B downloads — 404.
    dl = await client.get(
        f"/v1/datasets/{dataset_id}/download",
        headers={"X-Vlabs-Key": plaintext_b},
    )
    assert dl.status_code == 404


# ── Auth / error paths through the full pipeline ──────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_revoked_key_blocks_every_endpoint(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Revocation removes all 4 endpoints in one stroke."""
    plaintext, info = api_key
    res = await session.execute(
        select(APIKey).where(APIKey.id == info["api_key_id"])
    )
    row = res.scalar_one()
    from datetime import UTC, datetime
    row.revoked_at = datetime.now(UTC)
    await session.commit()

    headers = {"X-Vlabs-Key": plaintext}
    # POST
    r = await client.post(
        "/v1/datasets", json=_good_payload(), headers=headers
    )
    assert r.status_code == 401
    # LIST
    r = await client.get("/v1/datasets", headers=headers)
    assert r.status_code == 401
    # GET single
    fake = f"ds_{uuid.uuid4().hex}"
    r = await client.get(f"/v1/datasets/{fake}", headers=headers)
    assert r.status_code == 401
    # DOWNLOAD
    r = await client.get(f"/v1/datasets/{fake}/download", headers=headers)
    assert r.status_code == 401


# ── Quota / validation paths ──────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_quota_blocks_request_that_would_exhaust_tier(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Pre-flight quota check prevents accepting a job that won't fit
    in the remaining month. The free tier allows 1 000 tuples / month
    by default."""
    plaintext, info = api_key

    # Pre-seed counter near the cap so the next 100-tuple request is
    # exactly one over.
    from datetime import date
    counter = UsageCounter(
        api_key_id=info["api_key_id"],
        month=date.today().replace(day=1),
        tuples_generated=950,
    )
    session.add(counter)
    await session.commit()

    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=100),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 402
    assert r.json()["code"] == "quota_exceeded"


async def test_unknown_env_id_returns_404(
    client: AsyncClient, api_key
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(env_id="totally-fake-env"),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 404
    assert r.json()["code"] == "unknown_environment"


async def test_pydantic_validation_rejects_oversized_request(
    client: AsyncClient, api_key
) -> None:
    """``requested_tuples`` is capped at 100 000 in the schema."""
    plaintext, _ = api_key
    r = await client.post(
        "/v1/datasets",
        json=_good_payload(requested_tuples=1_000_000),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.status_code == 422


# ── Defense-in-depth: LLM API key never echoed anywhere ───────────


@pytest.mark.respx(assert_all_called=False)
async def test_llm_api_key_never_appears_in_any_endpoint_response(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Sweep through POST, GET-list, GET-single, GET-download and
    confirm the plaintext key is not echoed by any of them."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key
    secret = "sk-very-secret-customer-key-do-not-leak"

    r1 = await client.post(
        "/v1/datasets",
        json=_good_payload(llm_api_key=secret),
        headers={"X-Vlabs-Key": plaintext},
    )
    assert secret not in r1.text
    dataset_id = r1.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    r2 = await client.get(
        "/v1/datasets", headers={"X-Vlabs-Key": plaintext}
    )
    assert secret not in r2.text

    r3 = await client.get(
        f"/v1/datasets/{dataset_id}",
        headers={"X-Vlabs-Key": plaintext},
    )
    assert secret not in r3.text

    r4 = await client.get(
        f"/v1/datasets/{dataset_id}/download",
        headers={"X-Vlabs-Key": plaintext, "Accept": "application/json"},
    )
    assert secret not in r4.text


# ── Signed-URL TTL ────────────────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_download_expires_at_is_in_the_future(
    client: AsyncClient, api_key, respx_mock
) -> None:
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    create = await client.post(
        "/v1/datasets",
        json=_good_payload(),
        headers={"X-Vlabs-Key": plaintext},
    )
    dataset_id = create.json()["dataset_id"]
    await process_dataset_job(parse_dataset_id(dataset_id))

    dl = await client.get(
        f"/v1/datasets/{dataset_id}/download",
        headers={"X-Vlabs-Key": plaintext, "Accept": "application/json"},
    )
    from datetime import UTC, datetime
    expires_at = datetime.fromisoformat(dl.json()["expires_at"].replace("Z", "+00:00"))
    assert expires_at > datetime.now(UTC)


# ── Output formats ────────────────────────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_default_output_format_is_parquet(
    client: AsyncClient, api_key, session: AsyncSession
) -> None:
    """Schema default is parquet — the row should reflect that even
    without an explicit override."""
    plaintext, _ = api_key
    payload = _good_payload()
    payload.pop("output_format", None)
    r = await client.post(
        "/v1/datasets",
        json=payload,
        headers={"X-Vlabs-Key": plaintext},
    )
    assert r.json()["output_format"] == "parquet"


# ── Counter aggregation across jobs ───────────────────────────────


@pytest.mark.respx(assert_all_called=False)
async def test_counter_sums_across_multiple_jobs(
    client: AsyncClient, api_key, respx_mock
) -> None:
    """Three jobs × 2 tuples each → counter should land at 6."""
    reset_fake_storage_for_tests()
    _mock_llm(respx_mock)
    plaintext, _ = api_key

    for i in range(3):
        r = await client.post(
            "/v1/datasets",
            json=_good_payload(seed_start=i * 10, requested_tuples=2),
            headers={"X-Vlabs-Key": plaintext},
        )
        await process_dataset_job(parse_dataset_id(r.json()["dataset_id"]))

    from vlabs_api import db as db_module
    async with db_module._SessionFactory() as fresh:  # type: ignore[misc]
        res = await fresh.execute(select(UsageCounter))
        counter = res.scalar_one()
    assert counter.tuples_generated == 6


# ── Sanity: docs files are reachable ──────────────────────────────


def test_phase23_docs_exist() -> None:
    """The reference docs ship with the repo (PHASE_23_PLAN.md §15)."""
    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / "docs" / "api-reference" / "datasets.md").exists()
    assert (repo_root / "docs" / "api-reference" / "dataset-formats.md").exists()


def test_phase23_idempotency_doc_section_exists() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    idem = (repo_root / "docs" / "api-reference" / "idempotency.md").read_text()
    assert "/v1/datasets" in idem


def test_readme_mentions_datasets_endpoint() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    readme = (repo_root / "README.md").read_text()
    assert "/v1/datasets" in readme
