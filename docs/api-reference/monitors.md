# `/v1/monitors` — Continuous capability monitoring (Phase 28)

> **Status:** v0.0.1-alpha. The `/v1/monitors/*` endpoint family ships
> in Phase 28 of the verifiable-labs-envs roadmap.

A *monitor* is a long-lived configuration that registers a customer's
model endpoint with Verifiable Labs and asks us to run
[`vlabs-audit`](../../tools/vlabs-audit) against it on a schedule.
Each scheduled fire produces a *monitor run* — a per-fire row that
carries summary stats, a regression verdict, a PDF report, and (on
warning / regressed verdicts) one alert per configured channel.

## Endpoint surface

| Method | Path                                              | Purpose                                                                                          |
|--------|---------------------------------------------------|--------------------------------------------------------------------------------------------------|
| POST   | `/v1/monitors`                                    | Create a monitor.                                                                                |
| GET    | `/v1/monitors`                                    | List the caller's monitors (paginated).                                                          |
| GET    | `/v1/monitors/{monitor_id}`                       | Single-monitor detail.                                                                           |
| PATCH  | `/v1/monitors/{monitor_id}`                       | Partial update — name, cadence, env_subset, episodes_per_env, alert_channels, auth_token, status, rebaseline. |
| DELETE | `/v1/monitors/{monitor_id}`                       | Soft-delete (status → `failed`).                                                                 |
| POST   | `/v1/monitors/{monitor_id}/run`                   | Trigger an out-of-band audit (does NOT advance `next_run_at`).                                   |
| GET    | `/v1/monitors/{monitor_id}/runs`                  | Paginated run history.                                                                            |
| GET    | `/v1/monitors/{monitor_id}/runs/{monitor_run_id}` | Single run detail (carries the PDF presigned URL).                                                |

Auth: `X-Vlabs-Key` header (data-plane), same as
[`/v1/score`](score.md) and [`/v1/datasets`](datasets.md).

## `POST /v1/monitors` — create

Request body:

```json
{
  "name": "qwen-2.5-prod-2026Q2",
  "model_endpoint": "https://api.openai.com/v1",
  "model_name": "gpt-4o-mini",
  "auth_token": "sk-...",
  "cadence": "daily",
  "env_subset": ["math-algebra", "code-humaneval", "long-context-needle"],
  "episodes_per_env": 30,
  "alert_channels": [
    {"type": "email", "address": "ops@example.com"},
    {"type": "slack", "webhook_url": "https://hooks.slack.com/..."}
  ]
}
```

Response (201):

```json
{
  "monitor_id": "mon_...",
  "name": "qwen-2.5-prod-2026Q2",
  "status": "active",
  "cadence": "daily",
  "next_run_at": "2026-05-10T06:00:00Z",
  "auth_token_fingerprint": "a1b2c3d4",
  "projected_monthly_episodes": 2700,
  "tier_limit_episodes": 8100,
  "created_at": "2026-05-09T18:30:00Z"
}
```

### Tier caps

| Tier        | `monitors_max` | `monitor_envs_max` | `monitor_episodes_max` |
|-------------|----------------|---------------------|------------------------|
| free        | 1              | 1                   | 10                     |
| pro         | 3              | 3                   | 30                     |
| team        | 10             | 5                   | 50                     |
| enterprise  | unlimited      | up to 25            | up to 200              |

Create-time validation rejects requests whose `(cadence ×
env_subset_size × episodes_per_env)` projection exceeds the tier's
monthly episode ceiling (`402 monitor_tier_exceeded`).

### Auth-token security

The plaintext token is **only** accepted on `POST` / `PATCH`. The
response surface returns `auth_token_fingerprint` (first 8 hex chars
of SHA-256(token)) so the customer can verify "is this the key I
think it is?" without ever leaking the secret. At rest the token is
encrypted via Fernet (Phase 23.B `llm_key_crypto`); the encryption
key lives in the `VLABS_DATA_LLM_KEY_ENCRYPTION` Fly secret.

## `POST /v1/monitors/{id}/run` — trigger ad-hoc

Schedules a **manual** monitor run. Does NOT advance
`next_run_at` — the next scheduled fire still lands on its cadence.
Returns `202 Accepted` with the new `monitor_run_id`.

A run is rejected (409) if the monitor's `status != "active"`. Pause
first (`PATCH ... {"status": "paused"}`) to halt the schedule
without losing the configuration.

## Run lifecycle

```
queued → running → success | failed
```

- **queued** — created by the scheduler tick (or manual trigger);
  on Redis worker queue.
- **running** — picked up by a worker; `started_at` is set. Stale
  runs (>1h) are auto-reset to `failed` on next worker startup.
- **success** — episode batch completed; summary + verdict + PDF
  persisted. `cost_usd_estimate` records the customer's LLM spend
  (best-effort, from the endpoint's `usage` field).
- **failed** — terminal error. `error` carries a short reason
  (`endpoint_unreachable`, `decrypt`, `scheduler_lost_run`, etc.).

## Verdict semantics (D5-C + D5-A)

A successful run produces:

```json
{
  "regression_verdict": "ok" | "warning" | "regressed",
  "verdict_payload": {
    "conformal": {
      "current": 0.83, "baseline": 0.90, "delta_to_target": -0.07
    },
    "bootstrap": {
      "mean_delta": -0.12, "ci_low": -0.18, "ci_high": -0.06,
      "p_value": 0.012, "regressed": true
    },
    "per_env_breakdown": [...]
  }
}
```

**Combined verdict matrix** (per
[PHASE_28_PLAN.md §5 D5](../../PHASE_28_PLAN.md)):

| Conformal      | Bootstrap regressed | Combined    |
|----------------|----------------------|-------------|
| `regressed`    | (any)                | `regressed` |
| `warning`      | `True`               | `regressed` |
| `warning`      | `False`              | `warning`   |
| `ok`           | `True`               | `warning`   |
| `ok`           | `False`              | `ok`        |

`ok` runs **do not** dispatch alerts. `warning` and `regressed`
runs dispatch an email + (if configured) Slack notification with
the summary and a link to the dashboard run page.

## Baseline (D6-A)

The first successful run becomes the baseline. To rebaseline (after
shipping a new model version):

```bash
curl -X PATCH /v1/monitors/{id} \
  -H "X-Vlabs-Key: $VLABS_KEY" \
  -d '{"rebaseline": true}'
```

The next successful run will replace `baseline_run_id`.

## Retention (D11-A)

| Day     | State        | Storage                       |
|---------|--------------|-------------------------------|
| 0       | created      | Postgres + R2 (PDF)           |
| 90      | archived     | Postgres only (PDF deleted)   |
| 365     | hard-deleted | Postgres summary stats kept   |

Team / enterprise tier customers can extend retention via
`PATCH ... {"retention_days": 730}`.

## Cost model (D9-A)

Phase 28 uses the customer's own LLM key — Verifiable Labs **never**
pays for or sees plaintext API tokens. The `cost_usd_estimate` on
each run is best-effort from the endpoint's `usage` field; treat it
as approximate spend, not authoritative billing.

## See also

- [`PHASE_28_PLAN.md`](../../PHASE_28_PLAN.md) — full architecture
  decisions (D1-D11) + worker integration spec.
- [Continuous monitoring](../concepts/continuous-monitoring.md) —
  conceptual overview.
- [`/v1/datasets`](datasets.md) — async synthetic dataset generation
  (the Phase 23 sibling that shares the worker-pool pattern).
