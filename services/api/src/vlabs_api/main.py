"""FastAPI app entry point.

Exports ``app`` for ``uvicorn vlabs_api.main:app`` and ``run_dev`` for
the ``vlabs-api`` CLI script (``pip install -e .`` registers it).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from vlabs_api import __version__
from vlabs_api.config import get_settings
from vlabs_api.db import dispose_engine, init_engine
from vlabs_api.errors import APIError, to_problem_json
from vlabs_api.redis_client import aclose as redis_aclose
from vlabs_api.routes import (
    admin,
    audit,
    billing,
    calibrate,
    evaluate,
    health,
    keys,
    predict,
    usage,
    webhook,
)


def _init_sentry() -> None:
    """Initialise Sentry only when DSN is configured (skipped in tests)."""
    settings = get_settings()
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.vlabs_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        # FastAPI integration is auto-loaded via the [fastapi] extra.
        send_default_pii=False,
        release=__version__,
    )

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info(
        "vlabs_api.startup",
        version=__version__,
        environment=settings.vlabs_environment,
        log_level=settings.vlabs_log_level,
        billing_enabled=settings.vlabs_billing_enabled,
        ratelimit_backend="redis" if settings.upstash_redis_rest_url else "memory",
    )
    init_engine(settings.database_url)
    try:
        yield
    finally:
        await dispose_engine()
        await redis_aclose()
        log.info("vlabs_api.shutdown")


def create_app() -> FastAPI:
    _init_sentry()
    app = FastAPI(
        title="vlabs-api",
        version=__version__,
        description=(
            "Verifiable Labs Hosted Calibration API. Wraps vlabs-calibrate "
            "with auth, quotas, and audit history."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_exception_handler(APIError, to_problem_json)

    app.include_router(health.router)
    # Data plane — X-Vlabs-Key auth, tier-aware rate limit
    app.include_router(calibrate.router, prefix="/v1")
    app.include_router(predict.router, prefix="/v1")
    app.include_router(evaluate.router, prefix="/v1")
    app.include_router(audit.router, prefix="/v1")
    app.include_router(usage.router, prefix="/v1")
    # Management plane — Clerk JWT auth, billing + key issuance
    app.include_router(keys.router, prefix="/v1")
    app.include_router(billing.router, prefix="/v1")
    # Stripe webhook — signature-verified, no auth header
    app.include_router(webhook.router, prefix="/v1")
    # Admin plane — Clerk auth + allowlist
    app.include_router(admin.router, prefix="/v1")
    return app


app = create_app()


def run_dev() -> None:
    """Entry point registered as ``vlabs-api`` console script."""
    import uvicorn

    uvicorn.run(
        "vlabs_api.main:app",
        host="0.0.0.0",  # noqa: S104 — dev only; prod uses Fly's internal addressing
        port=8000,
        reload=True,
    )


__all__ = ["app", "create_app", "run_dev"]
