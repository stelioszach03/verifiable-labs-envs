"""FastAPI app entry point.

Exports ``app`` for ``uvicorn vlabs_api.main:app`` and ``run_dev`` for
the ``vlabs-api`` CLI script (``pip install -e .`` registers it).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded

from vlabs_api import __version__
from vlabs_api.config import get_settings
from vlabs_api.db import dispose_engine, init_engine
from vlabs_api.errors import APIError, to_problem_json
from vlabs_api.ratelimit import limiter, rate_limit_handler
from vlabs_api.routes import audit, calibrate, evaluate, health, predict, usage

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info(
        "vlabs_api.startup",
        version=__version__,
        environment=settings.vlabs_environment,
        log_level=settings.vlabs_log_level,
    )
    init_engine(settings.database_url)
    try:
        yield
    finally:
        await dispose_engine()
        log.info("vlabs_api.shutdown")


def create_app() -> FastAPI:
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
    app.state.limiter = limiter
    app.add_exception_handler(APIError, to_problem_json)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    app.include_router(health.router)
    app.include_router(calibrate.router, prefix="/v1")
    app.include_router(predict.router, prefix="/v1")
    app.include_router(evaluate.router, prefix="/v1")
    app.include_router(audit.router, prefix="/v1")
    app.include_router(usage.router, prefix="/v1")
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
