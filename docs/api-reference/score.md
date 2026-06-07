# `POST /v1/score`

**Per-call calibrated reward + audit row.** Phase 22.C.

The training-API hot path: customer transmits `(env_id, seed, completion)`,
service re-derives the instance, parses the completion via the env's
adapter, scores it, and persists an audit row. Returns the reward,
conformal interval, coverage guarantee, audit ID, and per-component
breakdown.

> The split-conformal coverage guarantee behind our calibrated reward intervals is machine-verified in Lean 4 (sorry-free, standard axioms only). The Python implementation is property-tested against the formal specification. See [`formal/VerifiableLabsFormal/ConformalCoverage.lean`](../../formal/VerifiableLabsFormal/ConformalCoverage.lean).

## Auth

`X-Vlabs-Key` header (data plane). Counts against the per-tier
`scores_per_month` quota — shared with [`POST /v1/instance`](instance.md).
Idempotent re-issues (see [Idempotency](idempotency.md)) do **not**
increment the counter.

## Request

```http
POST /v1/score HTTP/1.1
Content-Type: application/json
X-Vlabs-Key: vlk_<32-char>
X-Idempotency-Key: <optional, max 200 chars>

{
  "env_id": "math-algebra",
  "seed": 42,
  "completion": "{\"answer\": \"x**2 - 1\", \"confidence\": 0.9}",
  "idempotency_key": "<optional, alternative to header>",
  "difficulty_kwargs": {}
}
```

| Field              | Type    | Required | Notes                                          |
|--------------------|---------|----------|------------------------------------------------|
| `env_id`           | string  | yes      | One of the 13 registered envs.                 |
| `seed`             | int     | yes      | ≥ 0; deterministic instance generator.         |
| `completion`       | string  | yes      | Max 1 MB. Hashed (SHA-256) before storage.     |
| `idempotency_key`  | string? | no       | 24 h dedup window. Max 200 chars.              |
| `difficulty_kwargs`| object? | no       | Pass-through to env's `generate_instance`.     |

## Response — 200 OK

```json
{
  "reward": 1.0,
  "conformal_interval": [0.5, 1.0],
  "coverage_guarantee": 0.9,
  "audit_id": "aud_3f9c1e8a4b2d4f6c8e1a2b3c4d5e6f7a",
  "components_breakdown": {
    "format_valid": 1.0,
    "parse_valid": 1.0,
    "correct": 1.0
  },
  "env_version": "0.0.1",
  "latency_ms": 14
}
```

| Field                  | Type             | Notes                                                      |
|------------------------|------------------|------------------------------------------------------------|
| `reward`               | float            | Clamped to `[0, 1]`. NaN → 0.                              |
| `conformal_interval`   | `[float, float]` | `[low, high]` in `[0, 1]`, width = calibrated `q̂_α`.       |
| `coverage_guarantee`   | float            | `1 − α` from env hyperparams (default 0.9).                |
| `audit_id`             | string           | `aud_<32-char hex>`. Use with `/v1/score/audit/{id}`.      |
| `components_breakdown` | object           | Per-component sub-scores in `[0, 1]`.                      |
| `env_version`          | string           | Pinned per row; bump on reward-distribution changes.       |
| `latency_ms`           | int              | Server-measured score time (excludes network).             |

## Error responses

| Status | Code                  | When                                                    |
|--------|-----------------------|---------------------------------------------------------|
| 401    | `invalid_api_key`     | Missing / malformed / revoked `X-Vlabs-Key`.            |
| 402    | `quota_exceeded`      | Tier `scores_per_month` exhausted for the calendar month. |
| 404    | `unknown_environment` | `env_id` not in the registered env catalogue.           |
| 422    | (Pydantic)            | Body validation: missing field, wrong type, > 1 MB completion. |
| 429    | `rate_limited`        | Tier RPM exceeded (sliding 60 s window).                |

## Latency targets

| Env family                | p95         |
|---------------------------|-------------|
| symbolic-math             | < 200 ms    |
| inverse-problem (numeric) | < 500 ms    |

Hard timeout: **30 s per call**. On timeout, an audit row is still
written with `reward = 0` and the response carries `latency_ms` at
the timeout boundary.

## Concurrency

Imaging envs (mri-knee, lodopab-ct, sparse-fourier, phase-retrieval)
are gated by a per-env semaphore (size 4) to prevent FFT bursts from
starving the FastAPI event loop. Symbolic envs run under a much wider
semaphore (size 64); each call is sub-10 ms in practice.

## Audit trail

Every successful call writes a row to `audit_calls` with:
- `user_id`, `api_key_id`
- `env_id`, `env_version`, `seed`
- `completion_hash` (SHA-256 of the request body's `completion`)
- `reward`, `conformal_low`, `conformal_high`, `coverage`
- `components_json` (per-component scores)
- `latency_ms`, `idempotency_key`, `created_at`

The completion text itself is **never persisted** — see
[GDPR + audit-trail privacy guarantee](#privacy).

## Privacy

The completion is **never** stored in plaintext. Only the SHA-256 hash
goes into `audit_calls.completion_hash`. Customers can verify a row
matches their completion by re-hashing locally; nobody else can recover
the text. Retention: 90 days hot in Postgres, archived afterward.

## Idempotency

See [`idempotency.md`](idempotency.md). TL;DR — same
`(idempotency_key, user_id)` within 24 h returns the original audit
ID + reward without re-scoring or counter increment.

## Versioning

`env_version` is pinned per row from `verifiable_labs_envs.__version__`.
See [`env-versioning.md`](env-versioning.md) for the bump policy.

## Examples

### Python

```python
import httpx

r = httpx.post(
    "https://api.verifiable-labs.com/v1/score",
    headers={"X-Vlabs-Key": "vlk_..."},
    json={
        "env_id": "math-algebra",
        "seed": 42,
        "completion": '{"answer": "x**2 - 1", "confidence": 0.9}',
    },
)
print(r.json()["reward"])
```

### curl

```bash
curl -X POST https://api.verifiable-labs.com/v1/score \
  -H "X-Vlabs-Key: vlk_..." \
  -H "Content-Type: application/json" \
  -d '{"env_id":"math-algebra","seed":42,"completion":"{\"answer\":\"x**2 - 1\",\"confidence\":0.9}"}'
```
