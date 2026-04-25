"""Pydantic v2 request/response models for the Hosted Evaluation API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """``GET /v1/health`` payload."""

    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["0.1.0-alpha"])
    uptime_s: float = Field(..., ge=0.0)
    sessions_active: int = Field(..., ge=0)


class EnvironmentInfo(BaseModel):
    """One row of ``GET /v1/environments``."""

    id: str = Field(..., examples=["sparse-fourier-recovery"])
    qualified_id: str = Field(..., examples=["stelioszach/sparse-fourier-recovery"])
    domain: str = Field(..., examples=["compressed-sensing"])
    multi_turn: bool
    tool_use: bool
    description: str


class EnvironmentList(BaseModel):
    """``GET /v1/environments`` payload."""

    environments: list[EnvironmentInfo]
    count: int = Field(..., ge=0)


class CreateSessionRequest(BaseModel):
    """``POST /v1/sessions`` body."""

    env_id: str = Field(
        ...,
        description=(
            "Either a bare env id (``sparse-fourier-recovery``) or the "
            "Hub-qualified form (``stelioszach/sparse-fourier-recovery``). "
            "The owner prefix is stripped before lookup."
        ),
        examples=["stelioszach/sparse-fourier-recovery"],
    )
    seed: int = Field(default=0, ge=0)
    env_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Forwarded to ``load_environment`` (e.g. ``calibration_quantile``).",
    )


class CreateSessionResponse(BaseModel):
    """``POST /v1/sessions`` response. The ``observation`` is the raw
    instance payload the env's adapter would feed to a solver — the
    user is expected to consume it and submit a ``Prediction``-shaped
    answer back.
    """

    session_id: str
    env_id: str
    seed: int
    observation: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    expires_at: datetime


class SubmitRequest(BaseModel):
    """``POST /v1/sessions/{id}/submit`` body.

    Either ``answer_text`` (raw model output that the per-env adapter
    parses) **or** ``answer`` (already-shaped dict matching the env's
    ``Prediction`` dataclass via field-name) is accepted. ``answer_text``
    is the natural input for an LLM-driven evaluation; ``answer`` lets
    a classical solver client bypass adapter parsing.
    """

    model_config = ConfigDict(extra="forbid")

    answer_text: str | None = None
    answer: dict[str, Any] | None = None


class SubmitResponse(BaseModel):
    """``POST /v1/sessions/{id}/submit`` payload."""

    session_id: str
    reward: float
    components: dict[str, float]
    coverage: float | None = None
    parse_ok: bool
    complete: bool
    meta: dict[str, Any]


class SessionStateResponse(BaseModel):
    """``GET /v1/sessions/{id}`` payload."""

    session_id: str
    env_id: str
    seed: int
    created_at: datetime
    expires_at: datetime
    submissions: list[SubmitResponse]
    complete: bool


class LeaderboardRow(BaseModel):
    model: str
    n: int = Field(..., ge=0)
    mean_reward: float
    std_reward: float
    parse_fail_rate: float


class LeaderboardResponse(BaseModel):
    env_id: str
    rows: list[LeaderboardRow]
    sources: list[str] = Field(
        default_factory=list,
        description="CSV files that contributed rows to this aggregate.",
    )


class ErrorResponse(BaseModel):
    detail: str
