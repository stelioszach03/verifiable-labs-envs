"""Pydantic mirrors of the Hosted Evaluation API schemas.

These are **client-side** copies, deliberately decoupled from the
server-side `verifiable_labs_api.schemas` module so the SDK can be
``pip install``ed without pulling in the server's dependency tree
(fastapi / slowapi / structlog). The shapes are kept in lockstep with
the API; if a field changes server-side, mirror it here in the same PR.

Pydantic v2 is the only schema engine; mode = strict isn't enabled
because the server occasionally adds keys we want to accept (forward-
compat).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Environment(BaseModel):
    """One row of ``GET /v1/environments``."""

    model_config = ConfigDict(extra="allow")

    id: str
    qualified_id: str
    domain: str
    multi_turn: bool
    tool_use: bool
    description: str


class EnvironmentList(BaseModel):
    model_config = ConfigDict(extra="allow")

    environments: list[Environment]
    count: int


class HealthStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    version: str
    uptime_s: float
    sessions_active: int


class CreateSessionRequest(BaseModel):
    """Body for ``POST /v1/sessions``."""

    env_id: str
    seed: int = 0
    env_kwargs: dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    env_id: str
    seed: int
    observation: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime


class SubmitResponse(BaseModel):
    """Result of ``POST /v1/sessions/{id}/submit``."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    reward: float
    components: dict[str, float] = Field(default_factory=dict)
    coverage: float | None = None
    parse_ok: bool
    complete: bool
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    env_id: str
    seed: int
    created_at: datetime
    expires_at: datetime
    submissions: list[SubmitResponse] = Field(default_factory=list)
    complete: bool


class LeaderboardRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    n: int
    mean_reward: float
    std_reward: float
    parse_fail_rate: float


class LeaderboardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    env_id: str
    rows: list[LeaderboardRow] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    def top_models(self, n: int = 5) -> list[LeaderboardRow]:
        """Return the top-``n`` rows by ``mean_reward`` (descending)."""
        return sorted(self.rows, key=lambda r: -r.mean_reward)[:n]


__all__ = [
    "Environment",
    "EnvironmentList",
    "HealthStatus",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "SubmitResponse",
    "SessionState",
    "LeaderboardRow",
    "LeaderboardResponse",
]
