#!/bin/sh
# Container entrypoint — apply pending migrations, then hand off to uvicorn.
# Fly.io invokes this on every machine boot.

set -eu

echo "[entrypoint] starting vlabs-api"
echo "[entrypoint] env=${VLABS_ENVIRONMENT:-unset}  billing=${VLABS_BILLING_ENABLED:-unset}"

# Apply migrations idempotently. Alembic skips already-applied revisions.
echo "[entrypoint] running alembic upgrade head"
alembic upgrade head

# Hand off to the API server. Fly machines listen on the internal port set in
# fly.toml; we bind 0.0.0.0:8000 here. uvicorn workers default to 1 — Fly
# auto-scales by spinning up additional machines, not by adding workers
# inside one container.
echo "[entrypoint] launching uvicorn"
exec uvicorn vlabs_api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --access-log \
    --no-server-header
