# Multi-stage build for vlabs-api on Fly.io.
# Stage 1: build the api wheel + the verifiable-labs-envs wheel from
#          the in-repo source.
# Stage 2: runtime image (slim, non-root, no build tooling).
#
# Build context: REPO ROOT (not services/api/). Invoke from the repo
# root with:
#   flyctl deploy --app vlabs-api \
#                 --config services/api/fly.toml \
#                 --dockerfile services/api/deploy/Dockerfile

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
RUN pip install --upgrade pip wheel build

# 1. Build the verifiable-labs-envs wheel WITHOUT pulling its full
#    dependency tree (jax / scipy / scikit-image / matplotlib /
#    pillow / datasets / openai / tqdm / matplotlib are only needed
#    for env runtime — the api uses a thin slice covering
#    process_reward.{dataset,inference,segmentation} +
#    reward_distillation.stub_inference, all of which are
#    pure-Python + numpy. The api's own deps already install numpy.).
# LICENSE is required by hatchling because the root pyproject.toml
# declares ``license = { file = "LICENSE" }``. README.md satisfies
# the readme metadata field.
COPY pyproject.toml README.md LICENSE ./sdk/
COPY src ./sdk/src
RUN cd sdk && pip wheel --no-deps --wheel-dir /wheels '.'

# 2. Build the api wheel + its full transitive dependency tree.
#    pip wheel resolves vlabs-calibrate from PyPI; verifiable-labs-envs
#    is found in /wheels via --find-links.
COPY services/api/pyproject.toml services/api/README.md ./api/
COPY services/api/src ./api/src
RUN cd api && pip wheel \
        --find-links=/wheels \
        --wheel-dir /wheels \
        '.'

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
COPY services/api/migrations ./migrations
COPY services/api/alembic.ini ./alembic.ini
COPY services/api/deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

# Two-step install:
#   - vlabs-api with full dep resolution from /wheels.
#   - verifiable-labs-envs added with --no-deps so its env-runtime
#     extras (jax / scipy / scikit-image / matplotlib / pillow /
#     datasets / openai / tqdm) are NOT pulled in. The api's import
#     surface only touches the pure-Python / numpy slice; lazy
#     env-loading code paths (POST /v1/instance, POST /v1/score,
#     POST /v1/datasets, monitor runs) will still need the heavy deps
#     and are tracked as a follow-up image bake (Phase 32 hardening).
RUN pip install --no-index --find-links=/wheels vlabs-api \
    && pip install --no-index --find-links=/wheels --no-deps verifiable-labs-envs \
    && rm -rf /wheels \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R vlabs:vlabs /app

USER vlabs

EXPOSE 8000

# Liveness check from outside the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
