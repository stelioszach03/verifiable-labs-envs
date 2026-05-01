# Multi-stage build for vlabs-api on Fly.io.
# Stage 1: build wheels with all dependencies.
# Stage 2: runtime image (slim, non-root, no build tooling).

ARG PYTHON_VERSION=3.11

# ── Builder ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build tools needed by asyncpg + cryptography until they ship wheels.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# README.md is required by hatchling for metadata validation
# (pyproject.toml: readme = "README.md").
COPY pyproject.toml README.md ./
COPY src ./src

# Production image only needs runtime deps; the [dev] extra ships
# pgserver + ruff + mypy which are tens of MB of wheels we never use.
RUN pip install --upgrade pip wheel build \
    && pip wheel --wheel-dir /wheels '.'

# ── Runtime ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Minimal runtime libs: libpq for asyncpg, ca-certificates for HTTPS.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        libpq5 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 vlabs \
    && useradd --uid 1000 --gid vlabs --create-home --shell /bin/bash vlabs

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

# Install vlabs-api directly from the pre-built wheel by *name* (not '.').
# Going through `'.'` would re-trigger pyproject.toml's hatchling build path,
# but hatchling is in /wheels, not in the runtime env — and --no-index
# blocks PyPI fetches. `vlabs-api` resolves to the wheel produced by the
# builder stage, plus all its transitive deps which are also in /wheels.
RUN pip install --no-index --find-links=/wheels vlabs-api \
    && rm -rf /wheels \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R vlabs:vlabs /app

USER vlabs

EXPOSE 8000

# Liveness check from outside the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
