# Verifiable Labs API — deployment

Tier-1 Hosted Evaluation API, **v0.1.0-alpha**.

> The alpha runs without authentication and stores sessions in memory
> (single process). Don't put production traffic on it; treat the
> public endpoint as a developer playground.

## Local development

```bash
# 1. Install with the API extras
pip install -e ".[dev,api]"

# 2. Run uvicorn directly
uvicorn verifiable_labs_api.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Browse Swagger UI
open http://localhost:8000/docs

# 4. Health check
curl http://localhost:8000/v1/health
```

Or via Docker Compose (auto-reload, mounts the repo source):

```bash
docker compose -f deploy/api/docker-compose.yml up --build
```

## Deploy options

Three deploy targets are scaffolded; pick whichever the user wants
to flip live first. All three reuse the same `Dockerfile` so there
is no parallel image to maintain.

### Render.com (primary)

`render.yaml` is a Blueprint manifest. From the Render dashboard:

1. New → Blueprint.
2. Point at `https://github.com/stelioszach03/verifiable-labs-envs`.
3. Render auto-detects `deploy/api/render.yaml`.
4. Wait for the first build (~5 min).
5. Default URL: `https://verifiable-labs-api.onrender.com`.
6. Custom domain (later): uncomment the `domains:` section in
   `render.yaml`, add `api.verifiable-labs.com` CNAME at the
   registrar.

Free-tier caveats: instances sleep after 15 min of inactivity.
First request after sleep cold-starts in ~30 s. Acceptable for an
alpha; switch to a paid plan when traffic justifies it.

### Fly.io (backup)

`fly.toml` is the per-app config. From a workstation with `flyctl`:

```bash
fly launch --copy-config --no-deploy
fly deploy
```

Free tier: 3× shared-cpu-1× VMs across the org. Smaller cold-start
than Render.

### Self-hosted Docker / VPS

```bash
docker build -f deploy/api/Dockerfile -t verifiable-labs-api .
docker run -d \
    --name verifiable-labs-api \
    -p 8000:8000 \
    -e VL_API_RATE_LIMIT=30/minute \
    -e VL_API_CORS_ORIGINS="https://your-frontend.example" \
    verifiable-labs-api
```

For a real VPS deploy: front the container with Caddy or Nginx for
TLS termination; the container itself only speaks HTTP.

## Endpoints

All under `/v1/`. OpenAPI UI at `/docs` lists the full schema.

| method | path | purpose |
|---|---|---|
| GET | `/v1/health` | liveness + version + active session count |
| GET | `/v1/environments` | list all 10 envs with metadata |
| POST | `/v1/sessions` | start a session for an env + seed |
| POST | `/v1/sessions/{id}/submit` | submit `answer_text`, get score |
| GET | `/v1/sessions/{id}` | full session state (incl. submissions) |
| GET | `/v1/leaderboard?env_id=...` | aggregate benchmark results |

### Quick-start curl

```bash
# 1. Health
curl http://localhost:8000/v1/health

# 2. List envs
curl http://localhost:8000/v1/environments | jq '.environments[].id'

# 3. Start a session
session_id=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "content-type: application/json" \
  -d '{"env_id": "stelioszach/sparse-fourier-recovery", "seed": 0,
       "env_kwargs": {"calibration_quantile": 2.0}}' \
  | jq -r .session_id)
echo "session: $session_id"

# 4. Submit (truth answer for seed=0; replace with model output)
curl -X POST "http://localhost:8000/v1/sessions/$session_id/submit" \
  -H "content-type: application/json" \
  -d '{"answer_text": "{\"support_idx\": [...], \"support_amp_x1000\": [...]}"}'

# 5. Inspect session state
curl http://localhost:8000/v1/sessions/$session_id

# 6. Leaderboard
curl "http://localhost:8000/v1/leaderboard?env_id=sparse-fourier-recovery" | jq
```

### Python integration example

The Tier-1 SDK (`pip install verifiable-labs`, ships in Task 2)
wraps these endpoints. Until then, plain `httpx` works:

```python
import httpx

base = "http://localhost:8000/v1"
with httpx.Client(base_url=base, timeout=30) as c:
    r = c.post("/sessions", json={
        "env_id": "stelioszach/sparse-fourier-recovery",
        "seed": 0,
        "env_kwargs": {"calibration_quantile": 2.0},
    })
    sid = r.json()["session_id"]
    r = c.post(f"/sessions/{sid}/submit", json={"answer_text": my_model_output})
    print(r.json())
```

## Configuration

Environment variables consumed at startup:

| variable | default | meaning |
|---|---|---|
| `VL_API_RATE_LIMIT` | `30/minute` | slowapi rule, applied per remote address |
| `VL_API_CORS_ORIGINS` | `*` | CSV list, or `*` for any origin (alpha default) |

CORS is currently open (`*`); tighten for v0.2 once the production
frontend domain is decided.

## Limits + roadmap

- **No authentication** — anyone with the URL can submit. Rate-limited
  but not access-controlled. v0.2 (Tier 2) adds `X-VL-API-Key`.
- **Single-process session store** — sessions live in memory and don't
  survive restart. v0.2 swaps in Redis.
- **Multi-turn turn-dispatch** — the API records submissions for
  multi-turn envs but doesn't yet replay them through
  `env.run_rollout`. v0.2 routes turns through the env.
- **Structured `answer` payloads** — currently rejected with HTTP 422.
  v0.2 accepts a `Prediction`-shaped dict directly.

The full v0.1 → v0.2 plan lives in
[`/docs/SPRINT_GIGA_COMPLETE.md`](../../docs/SPRINT_GIGA_COMPLETE.md)
and the Tier 2 brief.
