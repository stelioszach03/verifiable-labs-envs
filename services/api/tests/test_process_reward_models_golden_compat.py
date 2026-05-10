"""Golden-shape tests for the Phase 30.E PRM endpoints.

Pins the JSON response shapes for ``/v1/process-reward-models/*`` so
a future schema change has to be intentional + reviewed. Pattern
mirrors ``test_reward_models_golden_compat.py``.

Also re-asserts the **22 prior endpoint surfaces** still respond
after adding /v1/process-reward-models/* (12 pre-Phase-29 + 5 from
Phase 29.E = 17, plus list/openapi smoke covering the whole set).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vlabs_api.db import ProcessRewardModel

DEFAULT_MODEL_ID = "vlabs-prm-distilled-qwen-1-5b-v0.1.0"


def _shape(value: Any, path: str = "") -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        out.add((path, "dict"))
        for k, v in value.items():
            sub = f"{path}.{k}" if path else k
            out |= _shape(v, sub)
    elif isinstance(value, list):
        out.add((path, "list"))
        if value:
            out |= _shape(value[0], f"{path}[]")
    elif value is None:
        out.add((path, "null"))
    else:
        out.add((path, type(value).__name__))
    return out


def _hdr(plaintext: str) -> dict[str, str]:
    return {"X-Vlabs-Key": plaintext}


async def _seed_default_model(session: AsyncSession) -> ProcessRewardModel:
    model = ProcessRewardModel(
        model_id=DEFAULT_MODEL_ID,
        name="Distilled PRM",
        family="distilled-qwen-1-5b",
        version="0.1.0",
        step_granularity="per_step",
        teacher_source="env+frontier",
        student_arch="Qwen2.5-1.5B-Instruct+lora",
        training_method="per-step-mse",
        status="available",
        eval_metrics={
            "processbench_overall": 0.62,
            "bon_lift_vs_phase29": 0.07,
            "aggregate_calibration_coverage": 0.91,
        },
        trained_at=datetime.now(UTC),
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


# ── /v1/process-reward-models list shape ──────────────────────────


async def test_compat_list_shape(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_default_model(session)
    r = await client.get(
        "/v1/process-reward-models", headers=_hdr(plaintext)
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("items", "list"),
        ("items[]", "dict"),
        ("items[].model_id", "str"),
        ("items[].family", "str"),
        ("items[].version", "str"),
        ("items[].status", "str"),
        ("items[].step_granularity", "str"),
        ("items[].created_at", "str"),
        ("items[].eval_summary", "dict"),
        ("total", "int"),
        ("limit", "int"),
        ("offset", "int"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── /v1/process-reward-models/{id} detail shape ───────────────────


async def test_compat_get_detail_shape(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_default_model(session)
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("model_id", "str"),
        ("family", "str"),
        ("version", "str"),
        ("status", "str"),
        ("step_granularity", "str"),
        ("teacher_source", "str"),
        ("student_arch", "str"),
        ("training_method", "str"),
        ("eval_summary", "dict"),
        ("eval_summary.processbench_overall", "float"),
        ("created_at", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── score response shape ──────────────────────────────────────────


async def test_compat_score_shape(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_default_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score",
        headers=_hdr(plaintext),
        json={
            "prompt": "p",
            "reasoning_trace": "Step 1: a.\nStep 2: b.",
        },
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("step_rewards", "list"),
        ("step_rewards[]", "float"),
        ("step_confidence_intervals", "list"),
        ("step_confidence_intervals[]", "list"),
        ("aggregate_reward", "float"),
        ("aggregate_confidence_interval", "list"),
        ("coverage_guarantee", "float"),
        ("step_count", "int"),
        ("model_id", "str"),
        ("schema_version", "str"),
        ("cache_hit", "bool"),
        ("latency_ms", "int"),
        ("audit_id", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── score/batch response shape ────────────────────────────────────


async def test_compat_score_batch_shape(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_default_model(session)
    r = await client.post(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/score/batch",
        headers=_hdr(plaintext),
        json={
            "items": [
                {
                    "prompt": "p",
                    "reasoning_trace": "Step 1: a.\nStep 2: b.",
                }
            ]
        },
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("items", "list"),
        ("items[]", "dict"),
        ("items[].step_rewards", "list"),
        ("items[].aggregate_reward", "float"),
        ("items[].audit_id", "str"),
        ("total", "int"),
        ("model_id", "str"),
        ("schema_version", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── evals shape ───────────────────────────────────────────────────


async def test_compat_evals_shape(
    client: AsyncClient, api_key, session: AsyncSession,
) -> None:
    plaintext, _ = api_key
    await _seed_default_model(session)
    r = await client.get(
        f"/v1/process-reward-models/{DEFAULT_MODEL_ID}/evals",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 200
    expected = {
        ("", "dict"),
        ("model_id", "str"),
        ("eval_summary", "dict"),
        ("held_out_envs", "dict"),
        ("processbench", "dict"),
        ("bon", "dict"),
        ("calibration", "dict"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── error response shape ──────────────────────────────────────────


async def test_compat_error_404_shape(
    client: AsyncClient, api_key,
) -> None:
    plaintext, _ = api_key
    r = await client.get(
        "/v1/process-reward-models/no-such-model-v0.1.0",
        headers=_hdr(plaintext),
    )
    assert r.status_code == 404
    expected = {
        ("", "dict"),
        ("type", "str"),
        ("title", "str"),
        ("status", "int"),
        ("code", "str"),
        ("detail", "str"),
    }
    missing = expected - _shape(r.json())
    assert not missing, f"missing keys: {missing}"


# ── 22 prior endpoint surfaces still mounted ──────────────────────


async def test_compat_prior_endpoints_still_mounted(
    client: AsyncClient,
) -> None:
    """Smoke check that the 22 pre-30.E endpoint paths still respond.

    12 pre-Phase-29 endpoints + 5 from Phase 29.E + 5 newly-added
    Phase 30.E endpoints = 22. Each path is hit without auth; we
    check that the response is not 404 (the route exists).
    """
    targets = [
        # 12 pre-Phase-29 surfaces.
        ("GET", "/health"),
        ("GET", "/v1/usage"),
        ("POST", "/v1/calibrate"),
        ("POST", "/v1/predict"),
        ("POST", "/v1/evaluate"),
        ("GET", "/v1/audit/cal_xxx"),
        ("POST", "/v1/instance"),
        ("POST", "/v1/score"),
        ("GET", "/v1/score/audit/aud_xxx"),
        ("GET", "/v1/datasets"),
        ("GET", "/v1/monitors"),
        ("POST", "/v1/keys"),
        # 5 from Phase 29.E.
        ("GET", "/v1/reward-models"),
        ("GET", "/v1/reward-models/some-id"),
        ("POST", "/v1/reward-models/some-id/score"),
        ("POST", "/v1/reward-models/some-id/score/batch"),
        ("GET", "/v1/reward-models/some-id/evals"),
        # 5 from Phase 30.E.
        ("GET", "/v1/process-reward-models"),
        ("GET", "/v1/process-reward-models/some-id"),
        ("POST", "/v1/process-reward-models/some-id/score"),
        ("POST", "/v1/process-reward-models/some-id/score/batch"),
        ("GET", "/v1/process-reward-models/some-id/evals"),
    ]
    for method, path in targets:
        if method == "GET":
            r = await client.get(path)
        else:
            r = await client.post(path, json={})
        assert r.status_code != 404, (
            f"prior endpoint {method} {path} returned 404 — was it dropped?"
        )


async def test_compat_openapi_includes_process_reward_models(
    client: AsyncClient,
) -> None:
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    for path in (
        "/v1/process-reward-models",
        "/v1/process-reward-models/{model_id}",
        "/v1/process-reward-models/{model_id}/score",
        "/v1/process-reward-models/{model_id}/score/batch",
        "/v1/process-reward-models/{model_id}/evals",
    ):
        assert path in paths, f"missing OpenAPI path {path}"


async def test_compat_22_prior_paths_in_openapi(
    client: AsyncClient,
) -> None:
    """The 22 prior + new endpoints are all documented in OpenAPI."""
    r = await client.get("/openapi.json")
    paths = r.json().get("paths", {})
    for path in (
        "/health",
        "/v1/usage",
        "/v1/calibrate",
        "/v1/predict",
        "/v1/evaluate",
        "/v1/instance",
        "/v1/score",
        "/v1/datasets",
        "/v1/monitors",
        "/v1/keys",
        "/v1/reward-models",
        "/v1/reward-models/{model_id}",
        "/v1/reward-models/{model_id}/score",
        "/v1/reward-models/{model_id}/score/batch",
        "/v1/reward-models/{model_id}/evals",
        "/v1/process-reward-models",
        "/v1/process-reward-models/{model_id}",
        "/v1/process-reward-models/{model_id}/score",
        "/v1/process-reward-models/{model_id}/score/batch",
        "/v1/process-reward-models/{model_id}/evals",
    ):
        assert path in paths, f"missing prior OpenAPI path {path}"
