# `/v1/reward-models` — Distilled reward model service (Phase 29)

> **Status:** v0.0.1-stub. The `/v1/reward-models/*` endpoint family
> ships in Phase 29.E of the verifiable-labs-envs roadmap. Until the
> trained student weights land in Phase 29.G, every score response
> carries `schema_version: "v0.1.0-stub"` and a deterministic
> `reward ∈ [0.4, 0.6]` derived from a SHA-256 hash of the
> (prompt, response) bytes. The endpoint contract is locked NOW so
> frontend + SDK integrations can land in parallel.

A *reward model* is a small open-weights student (Qwen2.5-1.5B with a
LoRA regression head per [PHASE_29_PLAN.md](../../PHASE_29_PLAN.md)
§5 D2-A / D3-A) trained to mimic the env-procedural reward signal
plus a frontier-judgment slice (D5-D ensemble). The serving layer
emits a calibrated scalar reward in `[0, 1]` plus a 90 % conformal
confidence interval (D10-A) — the same Layer 1 moat that backs
[`/v1/score`](score.md), extended one layer down.

## Endpoint surface

| Method | Path                                              | Purpose                                          |
|--------|---------------------------------------------------|--------------------------------------------------|
| GET    | `/v1/reward-models`                               | Paginated list (filter by `family`/`status`).   |
| GET    | `/v1/reward-models/{model_id}`                    | Single-model detail + eval card summary.        |
| POST   | `/v1/reward-models/{model_id}/score`              | Single-call scoring.                            |
| POST   | `/v1/reward-models/{model_id}/score/batch`        | Batch scoring (≤ 100 pairs/call).               |
| GET    | `/v1/reward-models/{model_id}/evals`              | Full eval card (held-out env breakdown +        |
|        |                                                   | RewardBench detail + calibration trace).         |

Auth: `X-Vlabs-Key` header (data-plane), same as
[`/v1/score`](score.md) and [`/v1/datasets`](datasets.md).

## `model_id` shape

`vlabs-reward-{family}-v{semver}` per locked D12-B versioning:

- `vlabs-reward-distilled-qwen-1-5b-v0.1.0` (Phase 29.G initial release)
- `vlabs-reward-distilled-qwen-1-5b-v0.1.1` (calibration tweak)
- `vlabs-reward-distilled-qwen-1-5b-v0.2.0` (D5 mix change)

A customer pinned to `v0.1.0` gets bit-deterministic outputs over the
lifetime of that version. New versions ship as new model_ids; the
customer migrates explicitly.

Lifecycle states (`reward_models.status`):

| Status      | Customer-visible? | Behaviour                                     |
|-------------|-------------------|-----------------------------------------------|
| `training`  | No (admin only)   | Hidden from list; 404 on detail / score.       |
| `available` | Yes               | Default-routable.                              |
| `deprecated`| Yes               | Still served; flagged in `status` field.       |
| `retired`   | No                | 404 on customer endpoints (admin can re-list). |

## `POST /v1/reward-models/{model_id}/score`

Request body:

```json
{
  "prompt": "What's the capital of France?",
  "response": "Paris.",
  "env_id": null,
  "schema_version": "v0.1.0"
}
```

Optional headers:

- `X-Vlabs-Cache: enable` — opt-in Redis cache (D11-C, default-off).
  Privacy posture: only SHA-256 hashes of prompt + response land in
  Redis; plaintext NEVER persists. Cache TTL is 1 hour. Cache hits
  return the same payload with `cache_hit: true` and ~95 % cost savings.
- `X-Idempotency-Key: <client-key>` — recorded on the audit row so
  the customer can detect retries.

Response (`200`):

```json
{
  "reward": 0.873,
  "confidence_interval": [0.789, 0.957],
  "coverage_guarantee": 0.90,
  "model_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
  "schema_version": "v0.1.0",
  "cache_hit": false,
  "latency_ms": 287,
  "audit_id": "rmr_a1b2c3d4e5f6789012345678901234ab"
}
```

Errors:

- `400 reward_model_invalid_request` — empty prompt/response or
  payload exceeds 1 MB.
- `404 reward_model_not_found` — unknown `model_id`, OR row exists
  in `training` status (admin-only). The same code is returned for
  both cases so admin state isn't leaked.
- `410 reward_model_retired` — model exists but is retired; SDKs
  should switch to a newer version.

## `POST /v1/reward-models/{model_id}/score/batch`

Up to 100 (prompt, response) pairs per call. Same headers as
single-score. Each item gets its own `audit_id` and `reward`. Idempotent
on `X-Idempotency-Key` — duplicate calls within the 24 h window
short-circuit to the prior response.

```json
{
  "items": [
    {"prompt": "...", "response": "...", "env_id": "math-algebra"},
    {"prompt": "...", "response": "..."}
  ],
  "schema_version": "v0.1.0"
}
```

## `GET /v1/reward-models`

```json
{
  "items": [
    {
      "model_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
      "family": "distilled-qwen-1-5b",
      "version": "0.1.0",
      "status": "available",
      "created_at": "2026-06-01T00:00:00Z",
      "eval_summary": {
        "rewardbench_overall": 0.71,
        "held_out_spearman_avg": 0.78,
        "calibration_coverage": 0.91
      }
    }
  ],
  "total": 1,
  "limit": 25,
  "offset": 0
}
```

Query params: `limit` (default 25, max 200), `offset`, `family`,
`status`. Sorted `created_at DESC` (newest first).

## `GET /v1/reward-models/{model_id}`

Full model record + eval summary. Fields:

- Identity: `model_id`, `name`, `family`, `version`.
- Provenance: `teacher_source` (e.g. `env+frontier`), `student_arch`
  (e.g. `Qwen2.5-1.5B-Instruct+lora`), `training_method`
  (e.g. `lora-mse`).
- Calibration: `conformal_quantile` (the locked split-conformal
  quantile, NULL until 29.F).
- Lifecycle: `status`, `created_at`, `trained_at`, `retired_at`.

## `GET /v1/reward-models/{model_id}/evals`

Full eval card — per-env Spearman + MAE + bias breakdown,
RewardBench-per-category accuracy, calibration drift trace. The
inner dicts evolve with each released model version (D12-B); SDKs
parse by key name, not position.

```json
{
  "model_id": "vlabs-reward-distilled-qwen-1-5b-v0.1.0",
  "eval_summary": {
    "rewardbench_overall": 0.71,
    "held_out_spearman_avg": 0.78,
    "calibration_coverage": 0.91
  },
  "held_out_envs": {
    "long-context-synthesis": {"spearman": 0.80, "mae": 0.06},
    "sql-multiturn":          {"spearman": 0.77, "mae": 0.08},
    "code-mini-repo":         {"spearman": 0.78, "mae": 0.07}
  },
  "rewardbench": {
    "chat": 0.72, "chat-hard": 0.68, "safety": 0.74, "reasoning": 0.69
  },
  "calibration": {
    "quantile": 0.087, "drift": 0.011, "n_calibration": 1000
  }
}
```

## Audit trail

Every score call writes a `reward_model_runs` row carrying:

- SHA-256 hashes of prompt + response (NEVER plaintext).
- Reward score + CI bounds + coverage guarantee.
- Cache hit flag, latency in ms, idempotency key (if supplied).

Customers can verify a row matches their own inputs by re-hashing
locally. Nobody else can recover the text — the same GDPR-aligned
posture as the Phase 22 [`audit_calls`](score.md) table.

## Pricing

Locked at:

- Per-call: $0.001–0.005 depending on token count.
- Batch (≥ 100 calls): $0.0005/call.
- Tier-included: free 1K/month, pro 100K, team 1M, enterprise
  per-contract.

Tier-cap enforcement lives in 29.F when the trained student is
billing-eligible; the 29.E stub server is free + uncounted.

## Latency budget

| Surface              | p50      | p95      |
|----------------------|----------|----------|
| Stub (29.E)          | < 50 ms  | < 100 ms |
| Trained (29.G live)  | < 500 ms | < 2 s    |
