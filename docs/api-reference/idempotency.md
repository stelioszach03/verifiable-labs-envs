# Idempotency

**`POST /v1/score` supports optional dedup via the `X-Idempotency-Key`
header.** Phase 22.C.

## Contract

| Property        | Value                                                       |
|-----------------|-------------------------------------------------------------|
| Header          | `X-Idempotency-Key: <client-supplied string>`               |
| Body alternative| `"idempotency_key": "<same string>"` in request JSON        |
| Max length      | 200 chars                                                   |
| Scope           | `(idempotency_key, user_id)` — different users do not collide |
| Window          | 24 h from first successful score call                       |

## Behaviour

### Re-issue inside the window

The original `audit_id` + reward + interval are returned verbatim.
**No new audit row** is written. **No usage-counter increment.** The
server is the single source of truth — even if the second request's
`completion` field differs, the response matches the first call.

### Re-issue outside the window

Treated as a fresh call. A new audit row is written. The old row is
deleted to make room for the new one (the partial unique index on
`(idempotency_key, user_id)` enforces a single row per active key).

### 5xx errors

If the server fails mid-score with a 5xx, **no audit row is written**
and the idempotency key is **not** consumed — clients can safely retry
the same key.

### Pydantic validation (4xx) before scoring

Request validation failures (e.g. negative seed, > 1 MB completion)
return 422 before scoring; the key is not consumed.

## Why use it

- **Retry safety.** Network blips during training are common; a retry
  with the same key won't double-count or double-write.
- **Replay determinism.** Your training loop's local logs can be
  reconciled against the server-side audit trail by joining on
  `(seed, idempotency_key)`.
- **Local debugging.** Set the idempotency key to a deterministic
  function of `(env_id, seed, training_step)` and the system becomes
  fully replayable.

## Worked example

```bash
# First call.
curl -X POST https://api.verifiable-labs.com/v1/score \
  -H "X-Vlabs-Key: vlk_..." \
  -H "X-Idempotency-Key: training-step-1234" \
  -d '{"env_id":"math-algebra","seed":42,"completion":"..."}'

# → returns audit_id "aud_AAA..." reward 0.7

# Same call, network retry — returns the SAME audit_id, no double-charge.
curl -X POST https://api.verifiable-labs.com/v1/score \
  -H "X-Vlabs-Key: vlk_..." \
  -H "X-Idempotency-Key: training-step-1234" \
  -d '{"env_id":"math-algebra","seed":42,"completion":"..."}'

# → audit_id "aud_AAA..." reward 0.7

# 25 hours later, same key — fresh row, fresh audit_id.
curl -X POST https://api.verifiable-labs.com/v1/score \
  -H "X-Vlabs-Key: vlk_..." \
  -H "X-Idempotency-Key: training-step-1234" \
  -d '{"env_id":"math-algebra","seed":42,"completion":"..."}'

# → audit_id "aud_BBB..." reward 0.7 (new row, counter incremented)
```

## What it does not do

- **It does not de-duplicate by completion content.** Two calls with
  the same completion but different (or absent) `idempotency_key`
  produce two distinct audit rows.
- **It does not work on `/v1/instance`.** Instance generation is
  stateless and idempotent by construction (same seed → same prompt);
  no server-side dedup is needed.
- **It does not span users.** Same key from two different API keys
  belonging to different users → two distinct audit rows.

## `/v1/datasets` { #datasets }

`POST /v1/datasets` supports the same dedup contract via a JSON-body
`idempotency_key` field (rather than the `X-Idempotency-Key` header).
The 24 h window covers an asynchronous lifecycle that may run for
hours, so the rules differ slightly:

- **In-window re-issue:** the original `dataset_id` + current `state`
  are returned. The customer's LLM API key is NOT re-validated and
  the worker queue is NOT re-enqueued — your earlier job (or its
  result) stands.
- **Out-of-window re-issue:** the stale row is deleted before the
  fresh insert (the partial unique index on `(idempotency_key,
  user_id)` enforces a single active row per key).
- **Quota:** in-window re-issues do **not** debit the
  `tuples_per_month` counter. The original job's per-tuple debits
  remain.

## See also

- [`score.md`](score.md) — full `/v1/score` request/response reference.
- [`datasets.md`](datasets.md) — `/v1/datasets` job lifecycle.
- [`env-versioning.md`](env-versioning.md) — when env_version changes
  break idempotency assumptions.
