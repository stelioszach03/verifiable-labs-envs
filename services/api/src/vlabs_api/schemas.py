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


# ── Phase 23 — vlabs-data dataset jobs ────────────────────────────


DatasetOutputFormat = Literal["parquet", "jsonl"]
DatasetJobState = Literal[
    "created",
    "queued",
    "running",
    "succeeded",
    "failed",
    "archived",
    "hard_deleted",
]


class DatasetCreateRequest(BaseModel):
    """``POST /v1/datasets`` request body (Phase 23.B).

    PHASE_23_PLAN.md §5.D1: customer brings their own LLM endpoint.
    The API key is encrypted at rest via ``pgp_sym_encrypt`` (pgcrypto)
    using the symmetric key from ``VLABS_DATA_LLM_KEY_ENCRYPTION``.

    The optional ``budget_usd_cap`` is a hard stop — the worker stops
    generation when ``budget_usd_spent >= budget_usd_cap`` and emits a
    ``state=succeeded`` row with ``generated_tuples < requested_tuples``
    (PHASE_23_PLAN.md §10).
    """

    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(min_length=1, max_length=128)
    requested_tuples: int = Field(ge=1, le=100_000)
    seed_start: int = Field(ge=0)

    # Customer-supplied LLM endpoint config (D1-B).
    llm_endpoint_url: str = Field(min_length=1, max_length=2_048)
    llm_api_key: str = Field(min_length=1, max_length=2_048)
    llm_model: str = Field(min_length=1, max_length=128)

    output_format: DatasetOutputFormat = "parquet"
    budget_usd_cap: float | None = Field(default=None, gt=0.0)
    idempotency_key: str | None = Field(default=None, max_length=200)


class DatasetCreateResponse(BaseModel):
    """``POST /v1/datasets`` response — returned immediately on enqueue.

    The job runs asynchronously; clients poll
    ``GET /v1/datasets/{dataset_id}`` for status.
    """

    dataset_id: str
    state: DatasetJobState
    requested_tuples: int
    seed_start: int
    seed_end: int
    output_format: DatasetOutputFormat
    env_version: str
    created_at: datetime


class DatasetJobResponse(BaseModel):
    """``GET /v1/datasets/{dataset_id}`` response body (Phase 23.D).

    The customer's LLM API key is NEVER returned (only the URL +
    model). Storage pointer fields populate as the job progresses
    through the lifecycle.
    """

    dataset_id: str
    env_id: str
    env_version: str
    requested_tuples: int
    generated_tuples: int
    seed_start: int
    seed_end: int
    llm_endpoint_url: str
    llm_model: str
    output_format: DatasetOutputFormat
    budget_usd_cap: float | None
    budget_usd_spent: float
    state: DatasetJobState
    # Aggregate stats (D9-C). NULL until completion.
    mean_reward: float | None
    std_reward: float | None
    p25_reward: float | None
    p50_reward: float | None
    p75_reward: float | None
    completion_success_rate: float | None
    # Storage integrity (populated on succeeded).
    storage_sha256: str | None
    storage_size_bytes: int | None
    error: str | None
    idempotency_key: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class DatasetJobSummary(BaseModel):
    """List-view row for ``GET /v1/datasets`` (Phase 23.D)."""

    dataset_id: str
    env_id: str
    env_version: str
    requested_tuples: int
    generated_tuples: int
    state: DatasetJobState
    created_at: datetime
    completed_at: datetime | None


class DatasetJobList(BaseModel):
    """``GET /v1/datasets`` response body (Phase 23.D)."""

    items: list[DatasetJobSummary]
    total: int
    limit: int
    offset: int


class DatasetDownloadResponse(BaseModel):
    """``GET /v1/datasets/{dataset_id}/download`` JSON variant (Phase 23.D).

    Default response is a 302 redirect to the signed URL. JSON variant
    requested via ``Accept: application/json`` returns the signed URL +
    integrity hash inline.
    """

    dataset_id: str
    download_url: str
    expires_at: datetime
    sha256: str
    size_bytes: int
    output_format: DatasetOutputFormat


# ── Phase 28 — continuous capability monitoring ─────────────────────


MonitorCadence = Literal["daily", "weekly", "monthly"]
MonitorStatus = Literal["active", "paused", "failed"]
MonitorRunStatus = Literal["queued", "running", "success", "failed"]
MonitorRunTrigger = Literal["scheduled", "manual"]
MonitorRegressionVerdict = Literal["ok", "warning", "regressed"]
MonitorAlertChannelType = Literal["email", "slack", "webhook"]


class MonitorAlertChannel(BaseModel):
    """One alert-dispatch channel attached to a monitor.

    The Slack webhook URL is encrypted at rest at the row layer
    (Phase 28.D); on the wire the customer supplies the plain URL,
    and the response surface returns ``url_fingerprint`` only (first
    8 hex chars of SHA-256). The same shape forward-extends for
    PagerDuty / generic-webhook channels in v0.0.2.
    """

    model_config = ConfigDict(extra="forbid")

    type: MonitorAlertChannelType
    address: str | None = Field(
        default=None,
        description="Email recipient address (type=email).",
        max_length=320,
    )
    webhook_url: str | None = Field(
        default=None,
        description="Slack webhook URL (type=slack).",
        max_length=2_048,
    )


class MonitorAlertChannelInfo(BaseModel):
    """Response-side projection — never includes raw secrets."""

    type: MonitorAlertChannelType
    address: str | None = None
    webhook_url_fingerprint: str | None = Field(
        default=None,
        description="First 8 hex chars of SHA-256(webhook_url).",
    )


class MonitorCreateRequest(BaseModel):
    """``POST /v1/monitors`` request body (Phase 28.B)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model_endpoint: str = Field(min_length=1, max_length=2_048)
    model_name: str = Field(min_length=1, max_length=128)
    auth_token: str = Field(min_length=1, max_length=4_096)
    cadence: MonitorCadence
    env_subset: list[str] = Field(min_length=1, max_length=25)
    episodes_per_env: int = Field(ge=1, le=200)
    alert_channels: list[MonitorAlertChannel] = Field(default_factory=list)


class MonitorUpdateRequest(BaseModel):
    """``PATCH /v1/monitors/{id}`` body — partial update (Phase 28.B)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    cadence: MonitorCadence | None = None
    env_subset: list[str] | None = Field(default=None, min_length=1, max_length=25)
    episodes_per_env: int | None = Field(default=None, ge=1, le=200)
    alert_channels: list[MonitorAlertChannel] | None = None
    auth_token: str | None = Field(default=None, min_length=1, max_length=4_096)
    status: MonitorStatus | None = None
    rebaseline: bool | None = None


class MonitorCreateResponse(BaseModel):
    """``POST /v1/monitors`` response (Phase 28.B)."""

    monitor_id: str
    name: str
    status: MonitorStatus
    cadence: MonitorCadence
    next_run_at: datetime
    auth_token_fingerprint: str
    projected_monthly_episodes: int
    tier_limit_episodes: int
    created_at: datetime


class MonitorResponse(BaseModel):
    """``GET /v1/monitors/{id}`` response (Phase 28.B)."""

    monitor_id: str
    name: str
    model_endpoint: str
    model_name: str
    auth_token_fingerprint: str
    cadence: MonitorCadence
    env_subset: list[str]
    episodes_per_env: int
    alert_channels: list[MonitorAlertChannelInfo]
    status: MonitorStatus
    retention_days: int
    baseline_run_id: str | None
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None
    next_run_at: datetime
    projected_monthly_episodes: int


class MonitorSummary(BaseModel):
    """List-view row for ``GET /v1/monitors`` (Phase 28.B)."""

    monitor_id: str
    name: str
    model_name: str
    cadence: MonitorCadence
    status: MonitorStatus
    env_subset: list[str]
    episodes_per_env: int
    last_run_at: datetime | None
    next_run_at: datetime
    created_at: datetime


class MonitorList(BaseModel):
    """``GET /v1/monitors`` response."""

    items: list[MonitorSummary]
    total: int
    limit: int
    offset: int


class MonitorRunSummary(BaseModel):
    """List-view row for ``GET /v1/monitors/{id}/runs`` (Phase 28.E).

    Surfaced in 28.B as the basic shape; the run lifecycle (status
    transitions, summary stats, verdict) lands as the worker
    integration in 28.C.
    """

    monitor_run_id: str
    monitor_id: str
    scheduled_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    status: MonitorRunStatus
    regression_verdict: MonitorRegressionVerdict | None
    trigger: MonitorRunTrigger
    cost_usd_estimate: float | None


class MonitorRunResponse(BaseModel):
    """``GET /v1/monitors/{id}/runs/{rid}`` response (Phase 28.E)."""

    monitor_run_id: str
    monitor_id: str
    scheduled_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    status: MonitorRunStatus
    summary_stats: dict[str, Any] | None
    regression_verdict: MonitorRegressionVerdict | None
    verdict_payload: dict[str, Any] | None
    pdf_url: str | None
    pdf_sha256: str | None
    cost_usd_estimate: float | None
    error: str | None
    trigger: MonitorRunTrigger


class MonitorRunList(BaseModel):
    items: list[MonitorRunSummary]
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


# ── V-Certified attestation programme (Phase 31.B) ──────────────────


AttestationScopeType = Literal["model", "deployment", "organization"]
AttestationTier = Literal["bronze", "silver", "gold"]
AttestationStatus = Literal[
    "draft",
    "submitted",
    "under_review",
    "approved",
    "revoked",
    "expired",
    "withdrawn",
]
AttestationCycle = Literal["annual", "continuous"]
AttestationArtifactKind = Literal[
    "training_doc",
    "audit_report",
    "monitor_record",
    "rm_record",
    "prm_record",
    "change_mgmt",
    "legal_signoff",
    "third_party_audit",
]
AttestationStandardName = Literal[
    "iso_42001",
    "nist_ai_rmf",
    "eu_ai_act",
    "soc2",
]
AttestationAuditDecision = Literal[
    "approve", "reject", "request_more", "revoke"
]
AttestationAuditorKind = Literal["self", "vlabs", "third_party"]


class AttestationCreateRequest(BaseModel):
    """``POST /v1/attestations`` request body.

    Customer creates a draft attestation; can later attach artifacts
    and submit. ``standards_requested`` is the subset of frameworks
    the customer wants crosswalk-aligned in the eventual report;
    must be a subset of the locked D8 enumeration.
    """

    model_config = ConfigDict(extra="forbid")

    organization: str = Field(min_length=1, max_length=200)
    scope_type: AttestationScopeType
    scope_subject: str = Field(min_length=1, max_length=500)
    tier: AttestationTier
    cycle: AttestationCycle
    standards_requested: list[AttestationStandardName] = Field(
        default_factory=list
    )


class AttestationPatchRequest(BaseModel):
    """``PATCH /v1/attestations/{id}`` request body.

    All fields optional — only set the ones being mutated. ``action``
    handles state transitions (``submit``, ``withdraw``).
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["submit", "withdraw"] | None = None
    organization: str | None = Field(default=None, min_length=1, max_length=200)
    scope_subject: str | None = Field(default=None, min_length=1, max_length=500)
    standards_requested: list[AttestationStandardName] | None = None


class AttestationStandardsAlignment(BaseModel):
    """Frozen-at-issuance crosswalk version snapshot (R1 mitigation)."""

    standards: list[AttestationStandardName] = Field(default_factory=list)
    crosswalk_version: str | None = None
    framework_versions: dict[str, str] = Field(default_factory=dict)


class AttestationInfo(BaseModel):
    """Full owner-facing attestation record."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    public_id: str
    organization: str
    scope_type: AttestationScopeType
    scope_subject: str
    tier: AttestationTier
    status: AttestationStatus
    cycle: AttestationCycle
    issued_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    revocation_reason: str | None = None
    cert_serial: str | None = None
    standards_alignment: AttestationStandardsAlignment = Field(
        default_factory=AttestationStandardsAlignment
    )
    artifact_count: int = 0
    created_at: datetime


class AttestationSummary(BaseModel):
    """Compact view used in paginated listings."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    public_id: str
    organization: str
    scope_type: AttestationScopeType
    scope_subject: str
    tier: AttestationTier
    status: AttestationStatus
    cycle: AttestationCycle
    issued_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime


class AttestationList(BaseModel):
    items: list[AttestationSummary]
    total: int
    limit: int
    offset: int


class AttestationArtifactRequest(BaseModel):
    """Metadata fields for ``POST /v1/attestations/{id}/artifacts``.

    The actual file bytes travel as the JSON ``content_b64`` field
    (base64-encoded). v0.0.1 uses a JSON body for simpler test
    fixturing; v0.0.2 may switch to multipart/form-data for
    large-file streaming.
    """

    model_config = ConfigDict(extra="forbid")

    kind: AttestationArtifactKind
    filename: str = Field(min_length=1, max_length=300)
    content_b64: str = Field(min_length=1)
    encrypted: bool = False


class AttestationArtifactInfo(BaseModel):
    """Returned by artifact-upload."""

    id: str
    attestation_id: str
    kind: AttestationArtifactKind
    storage_uri: str
    sha256_hash: str
    encrypted: bool
    size_bytes: int
    submitted_at: datetime


class AttestationRenewalRequest(BaseModel):
    """``POST /v1/attestations/{id}/renew`` body."""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str | None = Field(default=None, max_length=200)


class AttestationRenewalInfo(BaseModel):
    """Returned by renewal-initiation endpoint."""

    id: str
    attestation_id: str
    cycle_number: int
    initiated_at: datetime
    completed_at: datetime | None = None
    new_cert_serial: str | None = None


class AttestationRevokeRequest(BaseModel):
    """``DELETE /v1/attestations/{id}`` body — JSON, not query string."""

    model_config = ConfigDict(extra="forbid")

    revocation_reason: str = Field(min_length=1, max_length=1000)


class AttestationAuditEntry(BaseModel):
    """One row from the multi-party-audit trail (D12 / R6 transparency)."""

    id: str
    auditor_kind: AttestationAuditorKind
    auditor_label: str | None = None
    decision: AttestationAuditDecision
    audit_summary: dict[str, Any] = Field(default_factory=dict)
    decided_at: datetime


# ── Process reward model service (Phase 30.E) ────────────────────────


ProcessRewardModelStatus = Literal[
    "training", "available", "deprecated", "retired"
]
StepGranularity = Literal["per_step", "per_token", "per_stage"]


class ProcessRewardModelEvalSummary(BaseModel):
    """Compact per-PRM eval summary surfaced on the list endpoint."""

    processbench_overall: float | None = None
    bon_lift_vs_phase29: float | None = None
    aggregate_calibration_coverage: float | None = None


class ProcessRewardModelInfo(BaseModel):
    """Full PRM record surfaced by GET /v1/process-reward-models/{id}."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    family: str
    version: str
    base_rm_id: str | None = None
    step_granularity: StepGranularity
    teacher_source: str
    student_arch: str
    training_method: str
    status: ProcessRewardModelStatus
    aggregate_conformal_quantile: float | None = None
    eval_summary: ProcessRewardModelEvalSummary = Field(
        default_factory=ProcessRewardModelEvalSummary
    )
    created_at: datetime
    trained_at: datetime | None = None
    retired_at: datetime | None = None


class ProcessRewardModelSummary(BaseModel):
    """Compact view used in paginated listings."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    family: str
    version: str
    status: ProcessRewardModelStatus
    base_rm_id: str | None = None
    step_granularity: StepGranularity
    created_at: datetime
    eval_summary: ProcessRewardModelEvalSummary = Field(
        default_factory=ProcessRewardModelEvalSummary
    )


class ProcessRewardModelList(BaseModel):
    items: list[ProcessRewardModelSummary]
    total: int
    limit: int
    offset: int


class ProcessRewardScoreRequest(BaseModel):
    """``POST /v1/process-reward-models/{id}/score`` request body.

    ``reasoning_trace`` accepts either a single string (the server
    segments) or a pre-segmented list of step strings; the union is
    typed loosely via :class:`Any` to keep Pydantic happy across
    the two input shapes.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    reasoning_trace: str | list[str]
    env_id: str | None = None
    schema_version: str = "v0.1.0"
    with_step_rationale: bool = False


class ProcessRewardScoreResponse(BaseModel):
    """``POST /v1/process-reward-models/{id}/score`` response body."""

    model_config = ConfigDict(protected_namespaces=())

    step_rewards: list[float]
    step_confidence_intervals: list[tuple[float, float]]
    aggregate_reward: float = Field(ge=0.0, le=1.0)
    aggregate_confidence_interval: tuple[float, float]
    coverage_guarantee: float = Field(ge=0.0, le=1.0)
    step_count: int = Field(ge=1)
    model_id: str
    schema_version: str
    cache_hit: bool = False
    latency_ms: int = Field(ge=0)
    audit_id: str
    segmentation_warning: str | None = None


class ProcessRewardScoreBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    reasoning_trace: str | list[str]
    env_id: str | None = None


class ProcessRewardScoreBatchRequest(BaseModel):
    """``POST /v1/process-reward-models/{id}/score/batch`` body. Up to
    50 items per call (D7 — denser per-call shape than Phase 29's
    100-item cap)."""

    model_config = ConfigDict(extra="forbid")

    items: list[ProcessRewardScoreBatchItem] = Field(min_length=1, max_length=50)
    schema_version: str = "v0.1.0"


class ProcessRewardScoreBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[ProcessRewardScoreResponse]
    total: int
    model_id: str
    schema_version: str


class ProcessRewardEvalsResponse(BaseModel):
    """Full PRM eval card payload surfaced by
    ``GET /v1/process-reward-models/{id}/evals``."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    eval_summary: ProcessRewardModelEvalSummary
    held_out_envs: dict[str, Any] = Field(default_factory=dict)
    processbench: dict[str, Any] = Field(default_factory=dict)
    bon: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)


# ── Reward model service (Phase 29.E) ────────────────────────────────


RewardModelStatus = Literal["training", "available", "deprecated", "retired"]


class RewardModelEvalSummary(BaseModel):
    """Compact per-model eval summary surfaced on the list endpoint."""

    rewardbench_overall: float | None = None
    held_out_spearman_avg: float | None = None
    calibration_coverage: float | None = None


class RewardModelInfo(BaseModel):
    """Full reward-model record surfaced by GET /v1/reward-models/{id}."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    family: str
    version: str
    teacher_source: str
    student_arch: str
    training_method: str
    status: RewardModelStatus
    conformal_quantile: float | None = None
    eval_summary: RewardModelEvalSummary = Field(default_factory=RewardModelEvalSummary)
    created_at: datetime
    trained_at: datetime | None = None
    retired_at: datetime | None = None


class RewardModelSummary(BaseModel):
    """Compact view used in paginated listings."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    family: str
    version: str
    status: RewardModelStatus
    created_at: datetime
    eval_summary: RewardModelEvalSummary = Field(default_factory=RewardModelEvalSummary)


class RewardModelList(BaseModel):
    items: list[RewardModelSummary]
    total: int
    limit: int
    offset: int


class RewardScoreRequest(BaseModel):
    """``POST /v1/reward-models/{id}/score`` request body.

    ``schema_version`` lets clients gate against the response shape; the
    stub server in 29.E echoes the version with a ``-stub`` suffix when
    serving canned responses (so SDKs can detect that the trained
    student isn't online yet).
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    env_id: str | None = None
    schema_version: str = "v0.1.0"


class RewardScoreResponse(BaseModel):
    """``POST /v1/reward-models/{id}/score`` response body."""

    model_config = ConfigDict(protected_namespaces=())

    reward: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    coverage_guarantee: float = Field(ge=0.0, le=1.0)
    model_id: str
    schema_version: str
    cache_hit: bool = False
    latency_ms: int = Field(ge=0)
    audit_id: str


class RewardScoreBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    env_id: str | None = None


class RewardScoreBatchRequest(BaseModel):
    """``POST /v1/reward-models/{id}/score/batch`` body. Up to 100 items
    per call (D8). Idempotent on ``X-Idempotency-Key``."""

    model_config = ConfigDict(extra="forbid")

    items: list[RewardScoreBatchItem] = Field(min_length=1, max_length=100)
    schema_version: str = "v0.1.0"


class RewardScoreBatchResponse(BaseModel):
    """Same-order array of per-item responses + a summary."""

    model_config = ConfigDict(protected_namespaces=())

    items: list[RewardScoreResponse]
    total: int
    model_id: str
    schema_version: str


class RewardEvalsResponse(BaseModel):
    """Full eval card payload surfaced by
    ``GET /v1/reward-models/{id}/evals``.

    The shape is intentionally loose-typed: per-env, per-category, and
    calibration-trace dicts pass through as JSON blobs because they
    evolve with each released model version (D12-B). Customers parse
    by key name, not position.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    eval_summary: RewardModelEvalSummary
    held_out_envs: dict[str, Any] = Field(default_factory=dict)
    rewardbench: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)


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
    "DatasetOutputFormat",
    "DatasetJobState",
    "DatasetCreateRequest",
    "DatasetCreateResponse",
    "DatasetJobResponse",
    "DatasetJobSummary",
    "DatasetJobList",
    "MonitorCadence",
    "MonitorStatus",
    "MonitorRunStatus",
    "MonitorRunTrigger",
    "MonitorRegressionVerdict",
    "MonitorAlertChannelType",
    "MonitorAlertChannel",
    "MonitorAlertChannelInfo",
    "MonitorCreateRequest",
    "MonitorUpdateRequest",
    "MonitorCreateResponse",
    "MonitorResponse",
    "MonitorSummary",
    "MonitorList",
    "MonitorRunSummary",
    "MonitorRunResponse",
    "MonitorRunList",
    "DatasetDownloadResponse",
    "RewardModelStatus",
    "RewardModelEvalSummary",
    "RewardModelInfo",
    "RewardModelSummary",
    "RewardModelList",
    "RewardScoreRequest",
    "RewardScoreResponse",
    "RewardScoreBatchItem",
    "RewardScoreBatchRequest",
    "RewardScoreBatchResponse",
    "RewardEvalsResponse",
    "ProcessRewardModelStatus",
    "StepGranularity",
    "ProcessRewardModelEvalSummary",
    "ProcessRewardModelInfo",
    "ProcessRewardModelSummary",
    "ProcessRewardModelList",
    "ProcessRewardScoreRequest",
    "ProcessRewardScoreResponse",
    "ProcessRewardScoreBatchItem",
    "ProcessRewardScoreBatchRequest",
    "ProcessRewardScoreBatchResponse",
    "ProcessRewardEvalsResponse",
    "AttestationScopeType",
    "AttestationTier",
    "AttestationStatus",
    "AttestationCycle",
    "AttestationArtifactKind",
    "AttestationStandardName",
    "AttestationAuditDecision",
    "AttestationAuditorKind",
    "AttestationCreateRequest",
    "AttestationPatchRequest",
    "AttestationStandardsAlignment",
    "AttestationInfo",
    "AttestationSummary",
    "AttestationList",
    "AttestationArtifactRequest",
    "AttestationArtifactInfo",
    "AttestationRenewalRequest",
    "AttestationRenewalInfo",
    "AttestationRevokeRequest",
    "AttestationAuditEntry",
]
