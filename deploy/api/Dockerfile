# Verifiable Labs Hosted Evaluation API — Tier-1 v0.1.0-alpha.
#
# Builds a slim image that runs the FastAPI app under uvicorn on
# port 8000. Configure via env vars:
#   VL_API_RATE_LIMIT (default "30/minute")
#   VL_API_CORS_ORIGINS (default "*"; comma-separated list otherwise)
#
# Build:    docker build -f deploy/api/Dockerfile -t verifiable-labs-api .
# Run:      docker run -p 8000:8000 verifiable-labs-api
# Health:   curl http://localhost:8000/v1/health

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the metadata first to maximise Docker layer cache hits.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install runtime deps + the API extras.
RUN pip install --upgrade pip \
    && pip install -e ".[api]"

EXPOSE 8000

# Sensible production defaults; override with env vars at runtime.
ENV VL_API_RATE_LIMIT="30/minute" \
    VL_API_CORS_ORIGINS="*"

# Single worker is fine for the alpha — the sessions store is in-process.
# Switch to ``--workers 4`` once Redis-backed sessions land (Tier 2).
CMD ["uvicorn", "verifiable_labs_api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--access-log", "--log-level", "info"]
