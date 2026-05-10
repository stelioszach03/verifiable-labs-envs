"""Integration tests for the Phase 30.E PRM endpoints.

Exercises the 5 surfaces: list, detail, score, score/batch, evals.
Auth + idempotency + cache opt-in flow through the live ASGI client
+ pgserver fixtures defined in ``conftest.py``.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import (
    ProcessRewardModel,
    ProcessRewardModelRun,
    UsageCounter,
)

DEFAULT_MODEL_ID = "vlabs-prm-distilled-qwen-1-5b-v0.1.0"


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
    base_rm_id: uuid.UUID | None = None,
) -> ProcessRewardModel:
    model = ProcessRewardModel(
        model_id=model_id,
        name="Distilled PRM (Qwen 1.5B)",
        family=family,
        version=version,
        base_rm_id=base_rm_id,
        step_granularity="per_step",
        teacher_source="env+frontier",
        student_arch="Qwen2.5-1.5B-Instruct+lora",
        training_method="per-step-mse",
        status=status,
        eval_metrics=eval_metrics,
    )
    if status in ("available", "deprecated"):
        model.trained_at = datetime.now(UTC)
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


# ── GET /v1/process-reward-models ──────────────────────────────────


async def test_list_empty(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get("/v1/process-reward-models", headers=_hdr(plaintext))
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 25


async def test_list_excludes_training_and_retired(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, model_id="vlabs-prm-foo-v0.1.0", status="available")
    await _seed_model(session, model_id="vlabs-prm-bar-v0.1.0", status="training")
    await _seed_model(session, model_id="vlabs-prm-baz-v0.1.0", status="retired")
    await _seed_model(
        session, model_id="vlabs-prm-qux-v0.1.0", status="deprecated"
    )

    r = await client.get("/v1/process-reward-models", headers=_hdr(plaintext))
    assert r.status_code == 200
    statuses = {item["status"] for item in r.json()["items"]}
    assert "training" not in statuses
    assert "retired" not in statuses
    assert statuses == {"available", "deprecated"}


async def test_list_filters_by_family(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, model_id="vlabs-prm-foo-v0.1.0", family="foo")
    await _seed_model(session, model_id="vlabs-prm-bar-v0.1.0", family="bar")

    r = await client.get(
        "/v1/process-reward-models",
        headers=_hdr(plaintext),
        params={"family": "foo"},
    )
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["family"] == "foo"


async def test_list_pagination(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    for i in range(5):
        await _seed_model(
            session, model_id=f"vlabs-prm-foo-v0.{i}.0", version=f"0.{i}.0"
        )

    r = await client.get(
        "/v1/process-reward-models",
        headers=_hdr(plaintext),
        params={"limit": 2, "offset": 1},
    )
    body = r.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5


async def test_list_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/v1/process-reward-models")
    assert r.status_code == 401


# ── GET /v1/process-reward-models/{id} ─────────────────────────────


async def test_get_model_detail(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(
        session,
        eval_metrics={
            "processbench_overall": 0.62,
            "bon_lift_vs_phase29": 0.07,
            "aggregate_calibration_coverage": 0.91,
        },
    )
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["status"] == "available"
    assert body["step_granularity"] == "per_step"
    assert body["eval_summary"]["processbench_overall"] == pytest.approx(0.62)


async def test_get_model_detail_404_unknown(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/process-reward-models/no-such-model-v0.1.0",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 404
    assert r.json()["code"] == "process_reward_model_not_found"


async def test_get_model_detail_404_for_training_status(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, status="training")
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}", headers=_hdr(plaintext)
    )
    assert r.status_code == 404


async def test_get_model_detail_surfaces_base_rm_model_id(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """When base_rm_id is set, the detail response surfaces the parent
    Phase 29 RM by its public model_id."""
    from vlabs_api.db import RewardModel

    plaintext, _ = api_key
    rm = RewardModel(
        model_id="vlabs-reward-distilled-qwen-1-5b-v0.1.0",
        name="parent RM",
        family="distilled-qwen-1-5b",
        version="0.1.0",
        teacher_source="env",
        student_arch="qwen-1-5b+lora",
        training_method="lora-mse",
        status="available",
    )
    session.add(rm)
    await session.commit()
    await session.refresh(rm)
    await _seed_model(session, base_rm_id=rm.id)
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    assert r.json()["base_rm_id"] == rm.model_id


# ── POST /v1/process-reward-models/{id}/score ──────────────────────


async def test_score_returns_stub_payload(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={
            "prompt": "Solve 2x + 3 = 11",
            "reasoning_trace": "Step 1: subtract 3.\nStep 2: 2x = 8.\nStep 3: x = 4.",
        },
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["step_count"] == 3
    assert len(body["step_rewards"]) == 3
    assert len(body["step_confidence_intervals"]) == 3
    assert 0.0 <= body["aggregate_reward"] <= 1.0
    assert len(body["aggregate_confidence_interval"]) == 2
    assert body["coverage_guarantee"] == pytest.approx(0.9)
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["schema_version"].endswith("-stub")
    assert body["audit_id"].startswith("prr_")


async def test_score_accepts_pre_segmented_array(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={
            "prompt": "p",
            "reasoning_trace": ["First.", "Second.", "Third."],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["step_count"] == 3
    assert body["segmentation_warning"] is None


async def test_score_writes_audit_row_with_hashes_only(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={
            "prompt": "secret prompt",
            "reasoning_trace": "Step 1: secret first step.\nStep 2: secret second step.",
        },
    )

    from sqlalchemy import select

    rows = (
        await session.execute(select(ProcessRewardModelRun))
    ).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.prompt_hash and len(row.prompt_hash) == 64
    assert row.trace_hash and len(row.trace_hash) == 64
    assert "secret" not in row.prompt_hash
    assert "secret" not in row.trace_hash
    assert row.cache_hit is False
    assert row.step_count == 2


async def test_score_rejects_empty_prompt(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "", "reasoning_trace": "x"},
    )
    assert r.status_code == 422


async def test_score_404_unknown_model(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.post(
        "/v1/process-reward-models/no-such-model/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "reasoning_trace": "x"},
    )
    assert r.status_code == 404


async def test_score_404_retired_model(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Retired status hidden from customer; resolver returns 404."""
    plaintext, _ = api_key
    await _seed_model(session, status="retired")
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "reasoning_trace": "x"},
    )
    assert r.status_code == 404


async def test_score_records_idempotency_key(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    headers = {**_hdr(plaintext), "X-Idempotency-Key": "client-supplied-001"}
    await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=headers,
        json={"prompt": "p", "reasoning_trace": "Step 1: a.\nStep 2: b."},
    )
    from sqlalchemy import select

    rows = (
        await session.execute(select(ProcessRewardModelRun))
    ).scalars().all()
    assert rows[0].idempotency_key == "client-supplied-001"


async def test_score_cache_header_recognised(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    headers = {**_hdr(plaintext), "X-Vlabs-Cache": "enable"}
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=headers,
        json={"prompt": "p", "reasoning_trace": "Step 1: q.\nStep 2: a."},
    )
    assert r.status_code == 200
    assert isinstance(r.json()["cache_hit"], bool)


async def test_score_is_deterministic(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    payload = {
        "prompt": "deterministic-prompt",
        "reasoning_trace": "Step 1: deterministic.\nStep 2: deterministic too.",
    }
    r1 = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json=payload,
    )
    r2 = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json=payload,
    )
    assert r1.json()["step_rewards"] == r2.json()["step_rewards"]
    assert (
        r1.json()["aggregate_reward"]
        == r2.json()["aggregate_reward"]
    )


async def test_score_records_env_id_when_supplied(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={
            "prompt": "p",
            "reasoning_trace": "Step 1: a.\nStep 2: b.",
            "env_id": "math-algebra",
        },
    )
    from sqlalchemy import select

    rows = (
        await session.execute(select(ProcessRewardModelRun))
    ).scalars().all()
    assert rows[0].env_id == "math-algebra"


# ── POST /v1/process-reward-models/{id}/score/batch ────────────────


async def test_score_batch_basic(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    items = [
        {"prompt": f"p-{i}", "reasoning_trace": f"Step 1: a-{i}.\nStep 2: b-{i}."}
        for i in range(5)
    ]
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert len(body["items"]) == 5
    for item in body["items"]:
        assert item["audit_id"].startswith("prr_")
        assert item["model_id"] == DEFAULT_MODEL_ID
        assert item["step_count"] == 2


async def test_score_batch_writes_audit_per_item(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    items = [
        {"prompt": f"p-{i}", "reasoning_trace": f"Step 1: a-{i}.\nStep 2: b-{i}."}
        for i in range(3)
    ]
    await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    from sqlalchemy import select

    rows = (
        await session.execute(select(ProcessRewardModelRun))
    ).scalars().all()
    assert len(rows) == 3


async def test_score_batch_rejects_too_many(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """Plan §10: PRM batch caps at 50 (vs Phase 29's 100)."""
    plaintext, _ = api_key
    await _seed_model(session)
    items = [
        {"prompt": f"p-{i}", "reasoning_trace": "Step 1: a.\nStep 2: b."}
        for i in range(51)
    ]
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": items},
    )
    assert r.status_code == 422


async def test_score_batch_rejects_empty_items(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={"items": []},
    )
    assert r.status_code == 422


# ── GET /v1/process-reward-models/{id}/evals ───────────────────────


async def test_evals_returns_full_card(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(
        session,
        eval_metrics={
            "processbench_overall": 0.62,
            "bon_lift_vs_phase29": 0.07,
            "aggregate_calibration_coverage": 0.91,
            "held_out_envs": {"long-context-synthesis": {"spearman": 0.78}},
            "processbench": {"math": 0.62, "gsm8k": 0.71},
            "bon": {"prm_bon_lift_vs_rm": 0.07},
            "calibration": {
                "step_conformal_quantiles": {"range(0, 1)": 0.05},
                "aggregate_quantile": 0.087,
            },
        },
    )
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == DEFAULT_MODEL_ID
    assert body["eval_summary"]["processbench_overall"] == pytest.approx(0.62)
    assert "long-context-synthesis" in body["held_out_envs"]
    assert body["calibration"]["aggregate_quantile"] == pytest.approx(0.087)


async def test_evals_returns_empty_dicts_when_no_metrics(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, eval_metrics=None)
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["held_out_envs"] == {}
    assert body["processbench"] == {}
    assert body["bon"] == {}
    assert body["calibration"] == {}


async def test_evals_404_unknown(client: AsyncClient, api_key) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/process-reward-models/no-such-model/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 404


# ── auth + helpers ─────────────────────────────────────────────────


async def test_score_401_missing_key(client: AsyncClient) -> None:
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        json={"prompt": "p", "reasoning_trace": "x"},
    )
    assert r.status_code == 401


async def test_score_batch_401_missing_key(client: AsyncClient) -> None:
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        json={"items": [{"prompt": "p", "reasoning_trace": "x"}]},
    )
    assert r.status_code == 401


async def test_list_filter_with_training_returns_empty(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_model(session, status="training")
    r = await client.get(
        "/v1/process-reward-models",
        headers=_hdr(plaintext),
        params={"status": "training"},
    )
    assert r.status_code == 200
    assert r.json()["items"] == []


async def test_score_does_not_break_when_no_usage_counter_row(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    """30.E stub-mode score must succeed even without a usage_counters
    row — tier-cap enforcement is deferred to 30.G+."""
    plaintext, _ = api_key
    await _seed_model(session)
    from sqlalchemy import select

    rows = (await session.execute(select(UsageCounter))).scalars().all()
    assert rows == []
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={"prompt": "p", "reasoning_trace": "Step 1: a.\nStep 2: b."},
    )
    assert r.status_code == 200


# ── ID helpers ─────────────────────────────────────────────────────


def test_encode_process_reward_run_id_shape() -> None:
    from vlabs_api.ids import (
        encode_process_reward_run_id,
        parse_process_reward_run_id,
    )

    rid = uuid.uuid4()
    encoded = encode_process_reward_run_id(rid)
    assert encoded.startswith("prr_")
    assert len(encoded) == 4 + 32
    assert parse_process_reward_run_id(encoded) == rid


def test_parse_process_reward_run_id_invalid_raises() -> None:
    from vlabs_api.errors import ProcessRewardModelNotFound
    from vlabs_api.ids import parse_process_reward_run_id

    with pytest.raises(ProcessRewardModelNotFound):
        parse_process_reward_run_id("not-valid")
