"""``GET /health`` — liveness, no auth, no DB hit."""
from __future__ import annotations

from fastapi import APIRouter

from vlabs_api import __version__
from vlabs_api.config import get_settings
from vlabs_api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=get_settings().vlabs_environment,
    )
