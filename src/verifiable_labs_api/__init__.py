"""Verifiable Labs Hosted Evaluation API — v0.1.0-alpha.

REST wrapper around the ten ``verifiable_labs_envs`` environments.
Endpoints under ``/v1/``; OpenAPI UI auto-generated at ``/docs``.

Tier-1 alpha scope (see plan):

- No authentication. Public, rate-limited (30 req/min/IP).
- In-memory session store with TTL eviction (sessions expire after 1 h).
- CORS open during alpha; tighten in v0.2.
- ``/v1/health`` self-reports the ``v0.1.0-alpha`` version label so
  consumers know they are not on a stable release.

Construct the FastAPI app via :func:`create_app`; the module-level
``app`` instance is what ``uvicorn verifiable_labs_api.app:app``
imports for serving.
"""
from __future__ import annotations

__version__ = "0.1.0-alpha"

from verifiable_labs_api.app import create_app

__all__ = ["__version__", "create_app"]
