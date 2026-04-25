# REST API reference

Hosted Evaluation API, **v0.1.0-alpha**. Base URL:
`https://api.verifiable-labs.com` (placeholder — see
[deploy/api/README.md](https://github.com/stelioszach03/verifiable-labs-envs/blob/main/deploy/api/README.md)
for current live URL).

OpenAPI UI auto-generated at `{{base}}/docs`; this page is a
human-readable summary.

## Authentication

**v0.1: none.** The endpoint is open and rate-limited at 30 req/min/IP
via `slowapi`. CORS is set to `*` for the alpha. Production
deployments can opt in to a shared-secret header by setting `VL_API_KEY`
in the environment; clients send `X-VL-API-Key`.

**v0.2 plan.** Per-user keys, Redis-backed session store, tighter
CORS allowlist. Tracked in [Roadmap](../company/roadmap.md).

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/v1/health` | liveness + version + active session count |
| GET | `/v1/environments` | list all 10 envs with metadata |
| POST | `/v1/sessions` | start a session for an env + seed |
| GET | `/v1/sessions/{id}` | full session state (incl. submissions) |
| POST | `/v1/sessions/{id}/submit` | submit `answer_text`, get score |
| GET | `/v1/leaderboard?env_id=...` | aggregate benchmark results |

### `GET /v1/health`

```http
HTTP/1.1 200 OK
{
  "status": "ok",
  "version": "0.1.0-alpha",
  "uptime_s": 1247.3,
  "sessions_active": 12
}
```

### `GET /v1/environments`

Returns the 10 v0.1 envs with metadata. Use `qualified_id` (e.g.
`stelioszach/sparse-fourier-recovery`) when starting a session — bare
`id` works too but the qualified form is more explicit.

```http
HTTP/1.1 200 OK
{
  "environments": [
    {
      "id": "sparse-fourier-recovery",
      "qualified_id": "stelioszach/sparse-fourier-recovery",
      "domain": "compressed-sensing",
      "multi_turn": false,
      "tool_use": false,
      "description": "1D sparse Fourier recovery with OMP baseline."
    },
    …
  ],
  "count": 10
}
```

### `POST /v1/sessions`

Start a session. The response includes the **observation** the model
should respond to — this is the env-specific JSON shape that contains
`prompt_text`, `system_prompt`, and any per-instance numerics
(`mask`, `y`, `n`, `k`, etc.).

```http
POST /v1/sessions
{
  "env_id": "stelioszach/sparse-fourier-recovery",
  "seed": 42,
  "env_kwargs": {"calibration_quantile": 2.0}
}

HTTP/1.1 200 OK
{
  "session_id": "s-9a3b4c…",
  "env_id": "sparse-fourier-recovery",
  "seed": 42,
  "observation": {
    "prompt_text": "Recover the sparse signal …",
    "system_prompt": "You are an expert at sparse signal recovery …",
    "inputs": {"n": 256, "k": 10, "mask": [1, 5, 9, …], "y": [...]}
  },
  "metadata": {
    "qualified_id": "stelioszach/sparse-fourier-recovery",
    "adapter_attached": true,
    "ttl_seconds": 3600
  },
  "created_at": "2026-04-25T15:00:00+00:00",
  "expires_at": "2026-04-25T16:00:00+00:00"
}
```

`env_kwargs` is forwarded to the env's `load_environment(...)` factory.

Errors:

- `404` — env_id not in registry; response includes the
  `available_envs` list
- `422` — malformed body (Pydantic validation error)
- `429` — rate limit exceeded

### `POST /v1/sessions/{id}/submit`

Score a submission. The body contains the model's output as a string
(`answer_text`); the env's adapter parses it into a `Prediction` and
runs `env.score(prediction, instance)`.

```http
POST /v1/sessions/s-9a3b4c…/submit
{
  "answer_text": "{\"support_idx\": [12, 47, 91, …], \"support_amp_x1000\": [800, -1200, 450, …]}"
}

HTTP/1.1 200 OK
{
  "session_id": "s-9a3b4c…",
  "reward": 0.842,
  "components": {"nmse": 0.91, "support": 0.85, "conformal": 0.76},
  "coverage": 0.91,
  "parse_ok": true,
  "complete": true,
  "meta": {"weights": {"nmse": 0.4, "support": 0.3, "conformal": 0.3}}
}
```

For multi-turn envs, `complete = false` until the final turn or a
parse failure. The session-state endpoint shows the full submission
history.

Errors:

- `404` — session_id unknown or expired
- `422` — `answer_text` cannot be parsed by the env's adapter (the
  response includes the underlying `LLMSolverError`)
- `429` — rate limit

### `GET /v1/sessions/{id}`

Full session state — observation, all submissions, completion flag,
expiry. Useful for assembling multi-turn dialogues client-side.

```http
HTTP/1.1 200 OK
{
  "session_id": "s-9a3b4c…",
  "env_id": "sparse-fourier-recovery",
  "seed": 42,
  "created_at": "2026-04-25T15:00:00+00:00",
  "expires_at": "2026-04-25T16:00:00+00:00",
  "submissions": [<SubmitResponse>, …],
  "complete": true
}
```

### `GET /v1/leaderboard?env_id=...`

Aggregate model performance on a single env. Reads from
`results/llm_benchmark_v2.csv` and `complete_matrix_*.csv` (the
canonical published benchmarks).

```http
GET /v1/leaderboard?env_id=sparse-fourier-recovery

HTTP/1.1 200 OK
{
  "env_id": "sparse-fourier-recovery",
  "rows": [
    {"model": "anthropic/claude-haiku-4.5", "n": 12, "mean_reward": 0.554, "std_reward": 0.082, "parse_fail_rate": 0.05},
    {"model": "openai/gpt-5.4",            "n":  9, "mean_reward": 0.519, "std_reward": 0.073, "parse_fail_rate": 0.02},
    …
  ],
  "sources": ["complete_matrix_single_turn.csv", "llm_benchmark_v2.csv"]
}
```

## Limits

- 30 requests/minute/IP across all endpoints (configurable via
  `VL_API_RATE_LIMIT`).
- Sessions expire after 1 hour or when the in-memory store evicts
  them. Don't rely on long-lived state in v0.1.
- Multi-turn envs accept submissions, but turn dispatch is
  client-side. The SDK handles this transparently; raw HTTP clients
  must construct follow-up turns themselves.

## Source

[`src/verifiable_labs_api/`](https://github.com/stelioszach03/verifiable-labs-envs/tree/main/src/verifiable_labs_api).
The deployment IaC (Render, Fly.io, Docker) is at
[`deploy/api/`](https://github.com/stelioszach03/verifiable-labs-envs/tree/main/deploy/api).
