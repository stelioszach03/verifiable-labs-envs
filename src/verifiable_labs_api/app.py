"""FastAPI application for the Hosted Evaluation API (v0.1.0-alpha).

Endpoints (all under ``/v1``):

- ``GET  /v1/health``                    — liveness + version label
- ``GET  /v1/environments``              — list registered envs
- ``POST /v1/sessions``                  — start a new evaluation session
- ``POST /v1/sessions/{id}/submit``      — submit an answer, get score
- ``GET  /v1/sessions/{id}``             — full session state
- ``GET  /v1/leaderboard?env_id=...``    — aggregated benchmark numbers

OpenAPI UI is auto-generated at ``/docs``.

Tier-1 alpha scope:
- No auth. Public, rate-limited (30 req/min/IP). CORS open.
- In-memory session store (``SessionStore``); single-process only.
- Scoring routes through the per-env LLM adapter for ``answer_text``
  payloads. Structured ``answer`` payloads are reserved for v0.2.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from verifiable_labs_api import __version__
from verifiable_labs_api.leaderboard import aggregate_for_env
from verifiable_labs_api.registry import all_envs, normalize_env_id
from verifiable_labs_api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    EnvironmentInfo,
    EnvironmentList,
    HealthResponse,
    LeaderboardResponse,
    LeaderboardRow,
    SessionStateResponse,
    SubmitRequest,
    SubmitResponse,
)
from verifiable_labs_api.serialization import to_json_safe
from verifiable_labs_api.sessions import SessionStore, Submission
from verifiable_labs_envs import load_environment
from verifiable_labs_envs.solvers.llm_solver import LLMSolverError, get_adapter

# ─────────────────────────────────────────────────────────────────────
# Logging — JSON via structlog
# ─────────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger("verifiable_labs_api")


# ─────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────

DEFAULT_RATE_LIMIT = "30/minute"
DEFAULT_TTL_SECONDS = 3600


def create_app(
    *,
    session_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    rate_limit: str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Construct and return the FastAPI app.

    Parameters allow tests to inject smaller TTLs and looser rate
    limits without touching env vars. Production reads everything
    from ``VL_API_*`` env vars (see ``__main__`` block).
    """
    rate_limit = rate_limit or os.getenv("VL_API_RATE_LIMIT", DEFAULT_RATE_LIMIT)
    cors_origins = cors_origins or _default_cors_origins()
    store = SessionStore(ttl_seconds=session_ttl_seconds)
    limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("api.startup", version=__version__, rate_limit=rate_limit)
        yield
        log.info("api.shutdown", sessions_active=len(store))

    app = FastAPI(
        title="Verifiable Labs Evaluation API",
        version=__version__,
        description=(
            "REST wrapper around the Verifiable Labs scientific RL "
            "environments. **v0.1.0-alpha** — public + rate-limited; "
            "no authentication. See "
            "https://github.com/stelioszach03/verifiable-labs-envs."
        ),
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.limiter = limiter

    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # ---- Exception handlers ------------------------------------------------

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        log.warning("api.rate_limited",
                    client=get_remote_address(request),
                    detail=str(exc.detail))
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Rate limit exceeded: {exc.detail}"},
        )

    # ---- Routes ------------------------------------------------------------

    @app.get("/v1/health", response_model=HealthResponse, tags=["meta"])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            uptime_s=round(store.uptime_s, 3),
            sessions_active=len(store),
        )

    @app.get("/v1/environments", response_model=EnvironmentList, tags=["meta"])
    async def environments() -> EnvironmentList:
        metas = all_envs()
        rows = [EnvironmentInfo(**meta.__dict__) for meta in metas]
        return EnvironmentList(environments=rows, count=len(rows))

    @app.post(
        "/v1/sessions",
        response_model=CreateSessionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["sessions"],
    )
    async def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        bare = normalize_env_id(req.env_id)
        try:
            env = load_environment(bare, **req.env_kwargs)
        except KeyError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Unknown env: {req.env_id}"
            ) from exc
        except TypeError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"env_kwargs incompatible with {req.env_id}: {exc}",
            ) from exc

        try:
            instance = env.generate_instance(seed=req.seed)
        except Exception as exc:  # noqa: BLE001
            log.error("api.generate_failed", env_id=bare, seed=req.seed, error=str(exc))
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Failed to generate instance for {bare}: {exc}",
            ) from exc

        adapter = _safe_get_adapter(bare)
        observation = _build_observation(instance, adapter)
        metadata = {
            "env_id": bare,
            "qualified_id": f"stelioszach/{bare}",
            "adapter_attached": adapter is not None,
            "ttl_seconds": store._ttl,  # noqa: SLF001 — exposed by design
        }
        session = store.make_session(
            env_id=bare,
            seed=req.seed,
            instance=instance,
            env=env,
            metadata=metadata,
        )
        log.info("api.session_created",
                 session_id=session.session_id, env_id=bare, seed=req.seed)
        return CreateSessionResponse(
            session_id=session.session_id,
            env_id=bare,
            seed=req.seed,
            observation=observation,
            metadata=metadata,
            created_at=session.created_at,
            expires_at=session.expires_at,
        )

    @app.post(
        "/v1/sessions/{session_id}/submit",
        response_model=SubmitResponse,
        tags=["sessions"],
    )
    async def submit(session_id: str, req: SubmitRequest) -> SubmitResponse:
        try:
            session = store.get(session_id)
        except KeyError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Session not found or expired: {session_id}",
            ) from exc

        if req.answer_text is None and req.answer is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Provide either 'answer_text' (LLM output) or 'answer' "
                "(structured prediction).",
            )
        if req.answer is not None and req.answer_text is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "Structured 'answer' submission is reserved for v0.2; "
                "use 'answer_text' in v0.1.",
            )

        adapter = _safe_get_adapter(session.env_id)
        if adapter is None:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"No adapter registered for env {session.env_id}.",
            )

        try:
            prediction = adapter.parse_response(req.answer_text, session.instance)
        except LLMSolverError as exc:
            sub = Submission(
                answer_text=req.answer_text,
                answer=req.answer,
                reward=0.0,
                components={},
                coverage=None,
                parse_ok=False,
                meta={"parse_error": str(exc)},
                submitted_at=datetime.now(UTC),
            )
            session.submissions.append(sub)
            return SubmitResponse(
                session_id=session.session_id,
                reward=0.0,
                components={},
                coverage=None,
                parse_ok=False,
                complete=False,
                meta={"parse_error": str(exc)},
            )

        try:
            scored = session.env.score(prediction, session.instance)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Scoring failed: {exc}",
            ) from exc

        coverage = scored.get("meta", {}).get("coverage")
        components = {k: float(v) for k, v in scored.get("components", {}).items()}
        sub = Submission(
            answer_text=req.answer_text,
            answer=req.answer,
            reward=float(scored["reward"]),
            components=components,
            coverage=coverage,
            parse_ok=True,
            meta=to_json_safe(scored.get("meta", {})),
            submitted_at=datetime.now(UTC),
        )
        session.submissions.append(sub)
        # Single-shot envs complete on first valid submission. Multi-turn
        # envs ignore ``complete`` for now (v0.2 will dispatch turn logic).
        session.complete = not session.env_id.endswith("-multiturn")
        log.info("api.submission",
                 session_id=session.session_id,
                 reward=sub.reward, parse_ok=sub.parse_ok)

        return SubmitResponse(
            session_id=session.session_id,
            reward=sub.reward,
            components=sub.components,
            coverage=sub.coverage,
            parse_ok=sub.parse_ok,
            complete=session.complete,
            meta=sub.meta,
        )

    @app.get(
        "/v1/sessions/{session_id}",
        response_model=SessionStateResponse,
        tags=["sessions"],
    )
    async def get_session(session_id: str) -> SessionStateResponse:
        try:
            session = store.get(session_id)
        except KeyError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Session not found or expired: {session_id}",
            ) from exc
        submissions = [
            SubmitResponse(
                session_id=session.session_id,
                reward=s.reward,
                components=s.components,
                coverage=s.coverage,
                parse_ok=s.parse_ok,
                complete=session.complete,
                meta=s.meta,
            )
            for s in session.submissions
        ]
        return SessionStateResponse(
            session_id=session.session_id,
            env_id=session.env_id,
            seed=session.seed,
            created_at=session.created_at,
            expires_at=session.expires_at,
            submissions=submissions,
            complete=session.complete,
        )

    @app.get(
        "/v1/leaderboard",
        response_model=LeaderboardResponse,
        tags=["meta"],
    )
    async def leaderboard(env_id: str = Query(..., min_length=1)):
        bare = normalize_env_id(env_id)
        if bare not in {meta.id for meta in all_envs()}:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Unknown env: {env_id}"
            )
        agg = aggregate_for_env(bare)
        return LeaderboardResponse(
            env_id=env_id,
            rows=[LeaderboardRow(**row) for row in agg["rows"]],
            sources=agg["sources"],
        )

    return app


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _default_cors_origins() -> list[str]:
    raw = os.getenv("VL_API_CORS_ORIGINS", "*")
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _safe_get_adapter(env_id: str):
    try:
        return get_adapter(env_id)
    except KeyError:
        # Importing the package registers all known adapters; if a
        # specific env id is missing it's a real bug, log and continue.
        return None


def _build_observation(instance: Any, adapter: Any) -> dict[str, Any]:
    """Compose the observation payload returned to clients."""
    inputs = {}
    if hasattr(instance, "as_inputs"):
        try:
            inputs = to_json_safe(instance.as_inputs())
        except Exception:  # noqa: BLE001
            inputs = {}
    payload = {"inputs": inputs}
    if adapter is not None:
        try:
            payload["prompt_text"] = adapter.build_user_prompt(instance)
            payload["system_prompt"] = adapter.system_prompt
        except Exception:  # noqa: BLE001 — adapter bug shouldn't 500 the request
            payload["prompt_text"] = None
    return payload


# Default app instance for ``uvicorn verifiable_labs_api.app:app``.
app = create_app()
