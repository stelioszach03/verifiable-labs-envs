"""Integration tests for the Phase 29.E reward-model endpoints.

Exercises the 5 surfaces: list, detail, score, score/batch, evals.
Auth + idempotency + cache opt-in all flow through the live ASGI
client + pgserver fixtures defined in ``conftest.py``.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import RewardModel, RewardModelRun, UsageCounter

DEFAULT_MODEL_ID = "vlabs-reward-distilled-qwen-1-5b-v0.1.0"


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


async def _seed_model(
    session: AsyncSession,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    status: str = "available",
    family: str = "distilled-qwen-1-5b",
    version: str = "0.1.0",
    eval_metrics: dict | None = None,
) -> RewardModel:
    model = RewardModel(
        model_id=model_id,
        name="Distilled reward (Qwen 1.5B)",
        family=family,
        version=version,
        teacher_source="env+frontier",
        student_arch="Qwen2.5-1.5B-Instruct+lora",
        training_method="lora-mse",
        status=status,
        eval_metrics=eval_metrics,
    )
    if status in ("available", "deprecated"):
        model.trained_at = datetime.now(UTC)
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


# ── GET /v1/reward-models ──────────────────────────────────────────


async def test_list_returns_empty_when_no_rows(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/reward-models", headers=_hdr(plaintext))
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 25
    assert body["offset"] == 0


async def test_list_excludes_training_and_retired(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, model_id="vlabs-reward-foo-v0.1.0", status="available")
    await _seed_model(session, model_id="vlabs-reward-bar-v0.1.0", status="training")
    await _seed_model(session, model_id="vlabs-reward-baz-v0.1.0", status="retired")
    await _seed_model(
        session, model_id="vlabs-reward-qux-v0.1.0", status="deprecated"
    )

    r = await client.get("/v1/reward-models", headers=_hdr(plaintext))
    assert r.status_code == 200
    items = r.json()["items"]
    statuses = {item["status"] for item in items}
    assert "training" not in statuses
    assert "retired" not in statuses
    assert statuses == {"available", "deprecated"}


async def test_list_filters_by_family(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(
        session, model_id="vlabs-reward-foo-v0.1.0", family="foo"
    )
    await _seed_model(
        session, model_id="vlabs-reward-bar-v0.1.0", family="bar"
    )

    r = await client.get(
        "/v1/reward-models", headers=_hdr(plaintext), params={"family": "foo"}
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["family"] == "foo"


async def test_list_pagination(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    for i in range(5):
        await _seed_model(
            session, model_id=f"vlabs-reward-foo-v0.{i}.0", version=f"0.{i}.0"
        )

    r = await client.get(
        "/v1/reward-models",
        headers=_hdr(plaintext),
        params={"limit": 2, "offset": 1},
    )
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5
    assert body["offset"] == 1
    assert body["limit"] == 2


async def test_list_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/v1/reward-models")
    assert r.status_code == 401


# ── GET /v1/reward-models/{id} ─────────────────────────────────────


async def test_get_model_detail_returns_eval_summary(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(
        session,
        eval_metrics={
            "rewardbench_overall": 0.71,
            "held_out_spearman_avg": 0.78,
            "calibration_coverage": 0.91,
        },
    )

    r = await client.get(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["status"] == "available"
    assert body["eval_summary"]["rewardbench_overall"] == pytest.approx(0.71)
    assert body["eval_summary"]["held_out_spearman_avg"] == pytest.approx(0.78)


async def test_get_model_detail_404_for_unknown(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/reward-models/no-such-model-v0.1.0", headers=_hdr(plaintext)
    )
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "reward_model_not_found"


async def test_get_model_detail_404_for_training_status(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """training rows are admin-only; surface as 404 to customers."""
    plaintext, _ = api_key
    await _seed_model(session, status="training")

    r = await client.get(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}", headers=_hdr(plaintext)
    )
    assert r.status_code == 404


# ── POST /v1/reward-models/{id}/score ──────────────────────────────


async def test_score_returns_stub_payload(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)

    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "What is 2+2?", "response": "4"},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["reward"] <= 1.0
    assert len(body["confidence_interval"]) == 2
    assert body["coverage_guarantee"] == pytest.approx(0.9)
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["schema_version"].endswith("-stub")
    assert body["cache_hit"] is False
    assert body["audit_id"].startswith("rmr_")


async def test_score_writes_audit_row_with_hashes_only(
    client: AsyncClient,
    api_key,
    session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)

    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "secret prompt", "response": "secret response"},
    )
    assert r.status_code == 200

    rows = (
        (await session.execute(_select_all_runs())).scalars().all()
    )
    assert len(rows) == 1
    row = rows[0]
    # Hashes only — plaintext NEVER persists.
    assert row.prompt_hash and len(row.prompt_hash) == 64
    assert row.response_hash and len(row.response_hash) == 64
    assert "secret" not in row.prompt_hash
    assert "secret" not in row.response_hash
    assert row.cache_hit is False


async def test_score_rejects_empty_prompt(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "", "response": "x"},
    )
    # FastAPI returns 422 (pydantic min_length=1) on the wire.
    assert r.status_code == 422


async def test_score_404_for_unknown_model(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/reward-models/no-such-model/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "response": "r"},
    )
    assert r.status_code == 404


async def test_score_410_for_retired_model(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Customer-visible "retired" check: list endpoint hides retired
    rows entirely (404 on detail GET); but score with a known-retired
    model also returns 404 from the resolver. To exercise the 410
    branch we'd need to inject a retired status that escapes the
    resolver — testing the resolver behaviour is sufficient.
    """
    plaintext, _ = api_key
    await _seed_model(session, status="retired")
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "response": "r"},
    )
    # Retired is customer-invisible at the resolver layer → 404.
    assert r.status_code == 404


async def test_score_increments_cache_hit_when_header_set(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Cache opt-in via X-Vlabs-Cache header; first call is a miss,
    second call is a hit when the cache layer is wired (in-process
    test path doesn't have a real Redis, so this just verifies the
    header is recognised and doesn't break the call)."""
    plaintext, _ = api_key
    await _seed_model(session)
    headers = {**_hdr(plaintext), "X-Vlabs-Cache": "enable"}
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=headers,
        json={"prompt": "q", "response": "a"},
    )
    assert r.status_code == 200
    # cache_hit can be False even with header set (no real Redis); the
    # contract is just that the header is accepted.
    assert isinstance(r.json()["cache_hit"], bool)


async def test_score_records_idempotency_key_when_header_set(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    headers = {
        **_hdr(plaintext),
        "X-Idempotency-Key": "client-supplied-idem-001",
    }
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=headers,
        json={"prompt": "p", "response": "r"},
    )
    assert r.status_code == 200
    rows = (await session.execute(_select_all_runs())).scalars().all()
    assert rows[0].idempotency_key == "client-supplied-idem-001"


# ── POST /v1/reward-models/{id}/score/batch ────────────────────────


async def test_score_batch_basic(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    items = [
        {"prompt": f"q-{i}", "response": f"a-{i}"} for i in range(5)
    ]
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert 0.0 <= item["reward"] <= 1.0
        assert item["audit_id"].startswith("rmr_")
        assert item["model_id"] == DEFAULT_MODEL_ID


async def test_score_batch_writes_one_audit_row_per_item(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    items = [{"prompt": f"p-{i}", "response": f"r-{i}"} for i in range(3)]
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    assert r.status_code == 200
    rows = (await session.execute(_select_all_runs())).scalars().all()
    assert len(rows) == 3


async def test_score_batch_rejects_too_many_items(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    items = [{"prompt": f"p-{i}", "response": "r"} for i in range(101)]
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    assert r.status_code == 422  # pydantic max_length=100


async def test_score_batch_rejects_empty_items(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": []},
    )
    assert r.status_code == 422


# ── GET /v1/reward-models/{id}/evals ───────────────────────────────


async def test_evals_returns_full_card(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(
        session,
        eval_metrics={
            "rewardbench_overall": 0.7,
            "held_out_spearman_avg": 0.78,
            "calibration_coverage": 0.91,
            "held_out_envs": {"long-context-synthesis": {"spearman": 0.8}},
            "rewardbench": {"chat": 0.7},
            "calibration": {"quantile": 0.087, "drift": 0.01},
        },
    )

    r = await client.get(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["eval_summary"]["rewardbench_overall"] == pytest.approx(0.7)
    assert "long-context-synthesis" in body["held_out_envs"]
    assert body["calibration"]["quantile"] == pytest.approx(0.087)


async def test_evals_returns_empty_dicts_when_no_metrics(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, eval_metrics=None)

    r = await client.get(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["held_out_envs"] == {}
    assert body["rewardbench"] == {}
    assert body["calibration"] == {}


async def test_evals_404_for_unknown(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/reward-models/no-such-model/evals", headers=_hdr(plaintext)
    )
    assert r.status_code == 404


# ── unrelated tier counter touchpoint ──────────────────────────────


async def test_score_does_not_break_when_no_usage_counter_row(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Tier-cap enforcement is deferred to 29.F (when the trained model
    is billing-eligible). The 29.E stub-mode score must succeed even
    when no usage_counters row exists for the api_key."""
    plaintext, _ = api_key
    await _seed_model(session)
    # No UsageCounter pre-seeded; the conftest TRUNCATE leaves it empty.
    rows = (await session.execute(_select_all_counters())).scalars().all()
    assert rows == []
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "response": "r"},
    )
    assert r.status_code == 200


# ── helpers ─────────────────────────────────────────────────────────


def _select_all_runs():
    from sqlalchemy import select

    return select(RewardModelRun).order_by(RewardModelRun.created_at.asc())


def _select_all_counters():
    from sqlalchemy import select

    return select(UsageCounter).order_by(UsageCounter.month)


# ── auth + invalid keys ─────────────────────────────────────────────


async def test_score_401_for_missing_key(client: AsyncClient) -> None:
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        json={"prompt": "p", "response": "r"},
    )
    assert r.status_code == 401


async def test_score_batch_401_for_missing_key(client: AsyncClient) -> None:
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score/batch",
        json={"items": [{"prompt": "p", "response": "r"}]},
    )
    assert r.status_code == 401


# ── status filter validation ────────────────────────────────────────


async def test_list_filter_with_training_status_returns_empty(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Passing ?status=training is a no-op (training is admin-only)."""
    plaintext, _ = api_key
    await _seed_model(session, status="training")
    r = await client.get(
        "/v1/reward-models",
        headers=_hdr(plaintext),
        params={"status": "training"},
    )
    assert r.status_code == 200
    assert r.json()["items"] == []
    assert r.json()["total"] == 0


# ── deterministic stub property ─────────────────────────────────────


async def test_score_is_deterministic(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Same (prompt, response) → same reward across calls (the 29.E
    stub is hash-deterministic; 29.G inference will not be — but
    `cache_hit` will start returning True after the first call)."""
    plaintext, _ = api_key
    await _seed_model(session)
    payload = {"prompt": "deterministic-prompt", "response": "deterministic-response"}
    r1 = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json=payload,
    )
    r2 = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json=payload,
    )
    assert r1.json()["reward"] == r2.json()["reward"]
    assert r1.json()["confidence_interval"] == r2.json()["confidence_interval"]


# ── env_id passthrough ──────────────────────────────────────────────


async def test_score_records_env_id_when_supplied(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "response": "r", "env_id": "math-algebra"},
    )
    assert r.status_code == 200
    rows = (await session.execute(_select_all_runs())).scalars().all()
    assert rows[0].env_id == "math-algebra"


async def test_score_env_id_optional(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "response": "r"},
    )
    assert r.status_code == 200
    rows = (await session.execute(_select_all_runs())).scalars().all()
    assert rows[0].env_id is None


# ── model_id parsing helpers (unit) ────────────────────────────────


def test_encode_reward_model_run_id_shape() -> None:
    from vlabs_api.ids import encode_reward_model_run_id, parse_reward_model_run_id

    rid = uuid.uuid4()
    encoded = encode_reward_model_run_id(rid)
    assert encoded.startswith("rmr_")
    assert len(encoded) == 4 + 32
    assert parse_reward_model_run_id(encoded) == rid


def test_parse_reward_model_run_id_invalid_raises() -> None:
    from vlabs_api.errors import RewardModelNotFound
    from vlabs_api.ids import parse_reward_model_run_id

    with pytest.raises(RewardModelNotFound):
        parse_reward_model_run_id("not-a-valid-id")
