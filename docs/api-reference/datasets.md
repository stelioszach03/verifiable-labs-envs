# `/v1/datasets` — Synthetic-dataset jobs

**Async, calibrated synthetic data for offline RL & SFT.** Phase 23.

The `/v1/datasets` endpoints generate `(prompt, completion, reward,
components)` tuples on the customer's behalf, against a registered
verifiable env, using the customer's own LLM endpoint. Each tuple is
scored through the same conformal-calibrated reward function as the
[`POST /v1/score`](score.md) per-call API; the only differences are
that generation is **async** (jobs run on a worker pool, not in the
request) and the output ships as a single Parquet/JSONL file in
Cloudflare R2.

Use cases:
- Generate 100 k tuples for an offline DPO / SFT run without keeping
  a long HTTP connection open.
- Build a corpus of `(env_id, seed) → completion → reward` triples
  reproducible from the run config alone (every row carries the env
  version + reward components).
- Compare LLM endpoints on a fixed env at controlled cost — set
  `budget_usd_cap` and let the worker stop early.

## Endpoints

| Method | Path                                | Phase | What it does                              |
|--------|-------------------------------------|-------|-------------------------------------------|
| POST   | `/v1/datasets`                      | 23.B  | Enqueue a new job (returns immediately).  |
| GET    | `/v1/datasets`                      | 23.D  | Paginated list of caller's jobs.          |
| GET    | `/v1/datasets/{dataset_id}`         | 23.D  | Single job status + aggregate stats.      |
| GET    | `/v1/datasets/{dataset_id}/download`| 23.D  | 302 redirect to a presigned R2 URL.       |

All four are `X-Vlabs-Key`-authed and counted against the
`tuples_per_month` tier quota. Failed tuples (LLM transport error,
parse error, env-scoring crash) **do not** count against the quota.

## `POST /v1/datasets`

Create a job. The customer's LLM API key is encrypted at rest with
Fernet (symmetric); plaintext is **never** persisted.

### Request

```http
POST /v1/datasets HTTP/1.1
Content-Type: application/json
X-Vlabs-Key: vlk_<32-char>

{
  "env_id":           "math-algebra",
  "requested_tuples": 1000,
  "seed_start":       0,
  "llm_endpoint_url": "https://api.openai.com/v1",
  "llm_api_key":      "sk-...",
  "llm_model":        "gpt-4o-mini",
  "output_format":    "parquet",
  "budget_usd_cap":   2.00,
  "idempotency_key":  "my-job-2026-05-07"
}
```

| Field              | Type    | Required | Notes                                                                  |
|--------------------|---------|----------|------------------------------------------------------------------------|
| `env_id`           | string  | yes      | One of the registered envs.                                            |
| `requested_tuples` | int     | yes      | 1 ≤ N ≤ 100 000. Worker stops early if `budget_usd_cap` is hit first.  |
| `seed_start`       | int     | yes      | ≥ 0. Tuples cover seeds `[seed_start, seed_start + N - 1]`.            |
| `llm_endpoint_url` | string  | yes      | OpenAI-protocol Chat Completions endpoint (OpenRouter, vLLM, etc.).    |
| `llm_api_key`      | string  | yes      | Encrypted with Fernet at rest. Never returned by any endpoint.         |
| `llm_model`        | string  | yes      | Passed through to the customer endpoint as `model`.                    |
| `output_format`    | string? | no       | `"parquet"` (default) or `"jsonl"`.                                    |
| `budget_usd_cap`   | float?  | no       | Estimated USD spend cap. Strict — stops generation when exceeded.      |
| `idempotency_key`  | string? | no       | 24 h dedup window. In-window re-issues return the original `dataset_id`. |

### Response — 201 Created

```json
{
  "dataset_id":       "ds_8f3ef4ff7c3e496fbdfd944771403cbf",
  "state":            "queued",
  "requested_tuples": 1000,
  "seed_start":       0,
  "seed_end":         999,
  "output_format":    "parquet",
  "env_version":      "0.0.1",
  "created_at":       "2026-05-07T15:32:11Z"
}
```

The job is enqueued on Redis; the worker pool drains it asynchronously.
Poll `GET /v1/datasets/{dataset_id}` until `state` is `succeeded` or
`failed`.

### Errors

| Status | Code                  | When                                                       |
|--------|-----------------------|------------------------------------------------------------|
| 400    | (Pydantic)            | Body validation: missing field, wrong type, etc.           |
| 401    | `invalid_api_key`     | Missing / malformed / revoked `X-Vlabs-Key`.               |
| 402    | `quota_exceeded`      | `tier.tuples_per_month` would be exhausted by this request.|
| 404    | `unknown_environment` | `env_id` not in the registered env catalogue.              |
| 422    | (Pydantic)            | Field bounds (e.g. `requested_tuples > 100 000`).          |
| 429    | `rate_limited`        | Tier RPM exceeded.                                         |

## `GET /v1/datasets/{dataset_id}`

Single job detail. Returns the full lifecycle metadata; aggregate
reward stats populate once the worker writes the final row.

### Response — 200 OK (succeeded job)

```json
{
  "dataset_id":              "ds_8f3ef4ff7c3e496fbdfd944771403cbf",
  "env_id":                  "math-algebra",
  "env_version":             "0.0.1",
  "requested_tuples":        1000,
  "generated_tuples":        1000,
  "seed_start":              0,
  "seed_end":                999,
  "llm_endpoint_url":        "https://api.openai.com/v1",
  "llm_model":               "gpt-4o-mini",
  "output_format":           "parquet",
  "budget_usd_cap":          2.00,
  "budget_usd_spent":        1.42,
  "state":                   "succeeded",
  "mean_reward":             0.61,
  "std_reward":              0.21,
  "p25_reward":              0.50,
  "p50_reward":              0.65,
  "p75_reward":              0.80,
  "completion_success_rate": 0.94,
  "storage_sha256":          "a29f...",
  "storage_size_bytes":      823014,
  "error":                   null,
  "idempotency_key":         "my-job-2026-05-07",
  "created_at":              "2026-05-07T15:32:11Z",
  "started_at":              "2026-05-07T15:32:13Z",
  "completed_at":            "2026-05-07T15:41:47Z"
}
```

### Errors

| Status | Code                     | When                                            |
|--------|--------------------------|-------------------------------------------------|
| 401    | `invalid_api_key`        | Missing / malformed / revoked.                  |
| 404    | `dataset_job_not_found`  | Unknown id, malformed id, or owned by another user. |

The 404 surface is identical for "not yours" and "doesn't exist" — we
don't leak existence of other users' job ids.

## `GET /v1/datasets`

Paginated list. Sorted by `created_at DESC` (newest first).

### Query parameters

| Param    | Type   | Default | Notes                                                  |
|----------|--------|---------|--------------------------------------------------------|
| `limit`  | int    | 100     | 1 ≤ limit ≤ 500.                                       |
| `offset` | int    | 0       | ≥ 0.                                                   |
| `state`  | string | (any)   | Optional filter: `queued`/`running`/`succeeded`/...   |

### Response — 200 OK

```json
{
  "items": [
    {
      "dataset_id":       "ds_...",
      "env_id":           "math-algebra",
      "env_version":      "0.0.1",
      "requested_tuples": 1000,
      "generated_tuples": 1000,
      "state":            "succeeded",
      "created_at":       "2026-05-07T15:32:11Z",
      "completed_at":     "2026-05-07T15:41:47Z"
    }
  ],
  "total":  1,
  "limit":  100,
  "offset": 0
}
```

## `GET /v1/datasets/{dataset_id}/download`

Hand out a presigned URL for a succeeded job. Default response is a
**302 redirect** so a `curl -L` or browser-style client downloads
the file directly. Pass `Accept: application/json` to get the URL
inline alongside the SHA-256 + size — preferred for SDK use, where
you want to log the integrity hash before downloading.

### Response (302)

```http
HTTP/1.1 302 Found
Location: https://<r2-host>/<bucket>/<user>/<dataset>/parquet.parquet?X-Amz-Signature=...
```

### Response (JSON)

```json
{
  "dataset_id":     "ds_...",
  "download_url":   "https://<r2-host>/...?X-Amz-Signature=...",
  "expires_at":     "2026-05-07T16:41:47Z",
  "sha256":         "a29f...",
  "size_bytes":     823014,
  "output_format":  "parquet"
}
```

### Errors

| Status | Code                          | When                                      |
|--------|-------------------------------|-------------------------------------------|
| 401    | `invalid_api_key`             | Missing / malformed / revoked.            |
| 404    | `dataset_job_not_found`       | Unknown id, malformed id, or cross-user.  |
| 409    | `dataset_job_invalid_state`   | Job is not in `state=succeeded`.          |

URLs expire after **1 hour**. Re-fetch the endpoint to mint a new one.

## Output formats

See [`dataset-formats.md`](dataset-formats.md) for the on-disk layout
of Parquet vs JSONL — both share the same row schema.

## Idempotency

`POST /v1/datasets` honours an optional `idempotency_key` field on
the body (NOT the `X-Idempotency-Key` header used by `/v1/score` —
this is a JSON field because the same dedup window covers a
multi-hour async lifecycle). See
[`idempotency.md`](idempotency.md#datasets).

## Quota accounting

Each successfully scored tuple debits one unit from
`tier.tuples_per_month`. The deduction happens **per tuple**, not per
job — so a job that aborts halfway through (budget cap hit, customer
LLM 429s) only consumes quota for tuples that actually landed in the
output file. Failed tuples (LLM transport errors, parse errors) do
**not** count.
