"""Pydantic request and response models for every ``/v1/*`` endpoint.

Loose validation only (types, ranges, length bounds). Domain-level
validation (e.g. "trace requires uncertainty when nonconformity is
scaled_residual") lives in :mod:`vlabs_api.calibration` so the same
checks run regardless of how the data arrives.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NonconformityName = Literal["scaled_residual", "abs_residual", "binary"]


class CalibrationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_reward: float
    reference_reward: float
    uncertainty: float | None = Field(default=None, ge=0.0)


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    nonconformity: NonconformityName = "scaled_residual"
    traces: list[CalibrationTrace] = Field(min_length=2, max_length=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrateResponse(BaseModel):
    calibration_id: str
    alpha: float
    nonconformity: NonconformityName
    n_calibration: int
    quantile: float
    target_coverage: float
    nonconformity_stats: dict[str, float]
    created_at: datetime


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    predicted_reward: float
    uncertainty: float | None = Field(default=None, ge=0.0)


class PredictResponse(BaseModel):
    calibration_id: str
    predicted_reward: float
    sigma: float
    interval: tuple[float, float]
    quantile: float
    alpha: float
    target_coverage: float


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    traces: list[CalibrationTrace] = Field(min_length=1, max_length=1_000_000)
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class EvaluateResponse(BaseModel):
    calibration_id: str
    target_coverage: float
    empirical_coverage: float
    n: int
    n_in_interval: int
    interval_width_mean: float
    interval_width_median: float
    tolerance: float
    passes: bool
    nonconformity: dict[str, float]


class AuditEvaluation(BaseModel):
    n: int
    empirical_coverage: float
    passes: bool
    ts: datetime


class AuditResponse(BaseModel):
    calibration_id: str
    created_at: datetime
    alpha: float
    nonconformity: NonconformityName
    n_calibration: int
    quantile: float
    target_coverage: float
    nonconformity_stats: dict[str, float]
    metadata: dict[str, Any]
    evaluations: list[AuditEvaluation]


class TierQuota(BaseModel):
    traces_per_month: int
    rpm: int


class UsagePeriod(BaseModel):
    start: str
    end: str


class UsageCounts(BaseModel):
    traces: int
    calibrations: int
    evaluations: int
    predictions: int


class UsageRemaining(BaseModel):
    traces: int


class UsageResponse(BaseModel):
    tier: Literal["free", "pro", "team"]
    quota: TierQuota
    current_period: UsagePeriod
    usage: UsageCounts
    remaining: UsageRemaining


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: Literal["dev", "staging", "prod"]


# ── Phase 22 — Training API endpoints ───────────────────────────────


class InstanceRequest(BaseModel):
    """``POST /v1/instance`` request body."""

    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0)
    difficulty_kwargs: dict[str, Any] = Field(default_factory=dict)


class InstanceResponse(BaseModel):
    """``POST /v1/instance`` response body.

    ``prompt`` is the LLM-facing problem text rendered through the
    env's adapter (``adapter.build_user_prompt(instance)``). ``metadata``
    carries the env's public-input dict (``Instance.as_inputs()``)
    minus oracle fields. ``env_version`` pins the env catalogue
    revision used to render this instance — Phase 22.C ``/v1/score``
    re-checks the version on persistence.
    """

    instance_seed: int
    prompt: str
    metadata: dict[str, Any]
    env_version: str


class ScoreRequest(BaseModel):
    """``POST /v1/score`` request body.

    PHASE_22_PLAN.md §5.2: ``completion`` is the LLM's text response;
    server re-derives the instance from ``(env_id, seed)`` and
    calls ``adapter.parse_response(completion, instance)``. The
    completion is hashed (SHA-256) before storage — never persisted in
    plaintext (§5.3 GDPR guarantee).
    """

    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0)
    # 1 MB cap — adversarial completions rejected with 413 at the
    # FastAPI body-parse stage rather than reaching the env scorer.
    completion: str = Field(min_length=0, max_length=1_048_576)
    idempotency_key: str | None = Field(default=None, max_length=200)
    difficulty_kwargs: dict[str, Any] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    """``POST /v1/score`` response body."""

    reward: float
    conformal_interval: tuple[float, float]
    coverage_guarantee: float
    audit_id: str
    components_breakdown: dict[str, float]
    env_version: str
    latency_ms: int


class AuditCallResponse(BaseModel):
    """``GET /v1/score/audit/{audit_id}`` response body (Phase 22.D).

    Mirrors the fields on the ``audit_calls`` row. Completion text is
    NEVER returned — only its SHA-256 hash, per the GDPR guarantee in
    PHASE_22_PLAN.md §5.3.
    """

    audit_id: str
    env_id: str
    env_version: str
    seed: int
    completion_hash: str
    reward: float
    conformal_interval: tuple[float, float]
    coverage_guarantee: float
    components_breakdown: dict[str, float]
    latency_ms: int
    idempotency_key: str | None
    created_at: datetime


class AuditCallSummary(BaseModel):
    """List-view row — light enough to paginate at 100 per page."""

    audit_id: str
    env_id: str
    env_version: str
    reward: float
    latency_ms: int
    created_at: datetime


class AuditCallList(BaseModel):
    """``GET /v1/score/audit`` response body (Phase 22.D).

    Offset-pagination: clients page via ``?limit=N&offset=K``. ``total``
    is the count of rows owned by this user (cheap COUNT query against
    the ``audit_calls_user_idx`` index).
    """

    items: list[AuditCallSummary]
    total: int
    limit: int
    offset: int


# ── Stage B: billing + key management schemas ────────────────────────


PaidTier = Literal["pro", "team"]


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: PaidTier


class CheckoutResponse(BaseModel):
    url: str
    tier: PaidTier


class PortalResponse(BaseModel):
    url: str


class CreateAPIKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)


class APIKeyInfo(BaseModel):
    id: str
    prefix: str
    name: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class APIKeyCreated(APIKeyInfo):
    plaintext_key: str = Field(
        description="Returned ONCE on creation; never persisted in plaintext."
    )


class APIKeyList(BaseModel):
    items: list[APIKeyInfo]


class AdminDashboardCounts(BaseModel):
    users: int
    api_keys_active: int
    api_keys_revoked: int
    calibrations_total: int
    evaluations_total: int
    subscriptions_active: int


class AdminDashboardLastRun(BaseModel):
    calibration_id: str
    api_key_prefix: str
    n_calibration: int
    quantile: float
    created_at: datetime


class AdminDashboardResponse(BaseModel):
    """Aggregate stats served by GET /v1/admin/dashboard."""

    counts: AdminDashboardCounts
    most_recent_calibrations: list[AdminDashboardLastRun]
    billing_enabled: bool


__all__ = [
    "NonconformityName",
    "CalibrationTrace",
    "CalibrateRequest",
    "CalibrateResponse",
    "PredictRequest",
    "PredictResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "AuditEvaluation",
    "AuditResponse",
    "TierQuota",
    "UsagePeriod",
    "UsageCounts",
    "UsageRemaining",
    "UsageResponse",
    "HealthResponse",
    "PaidTier",
    "CheckoutRequest",
    "CheckoutResponse",
    "PortalResponse",
    "CreateAPIKeyRequest",
    "APIKeyInfo",
    "APIKeyCreated",
    "APIKeyList",
    "AdminDashboardCounts",
    "AdminDashboardLastRun",
    "AdminDashboardResponse",
    "InstanceRequest",
    "InstanceResponse",
    "ScoreRequest",
    "ScoreResponse",
    "AuditCallResponse",
    "AuditCallSummary",
    "AuditCallList",
]
