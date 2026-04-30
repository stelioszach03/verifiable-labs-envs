# Phase 16 — Hosted Calibration API (`vlabs-api`)

> Plan-only deliverable. After approval the work splits into 3 stages (A/B/C) shipped as 3 commits.

## 0. Pre-flight — observed state

| Check | Observed | Implication |
|---|---|---|
| HEAD on origin/main | `a645f994` (Phase 15.B CI integration) | Clean tip; Phase 15.A + 15.B both shipped. |
| Existing tests | 519 passing (519/525 incl. 6 skipped per Phase 15.B) | Hard guarantee for additivity. |
| `services/` directory | absent | Greenfield — no merge surface. |
| Local processes interfering | None on this WSL host | Phase E (Colab PID 48395) is on a different machine and out of reach by definition. |
| Working tree | dirty with M3/M4 untracked (`tests/training/`, `examples/training/`, etc.) | Phase 16 only stages files under `services/api/` and `services/landing/` — M3/M4 stays untouched. |
| `vlabs-calibrate` | v0.1.0a1 on PyPI; importable via `pip install vlabs-calibrate` | Pinned as **external dep** of the API service so the SaaS layer reproduces the same SDK end users get. |

---

## 1. Executive summary

1. **The hosted API never executes user code.** It accepts pre-computed `(predicted_reward, reference_reward, uncertainty)` triples and runs the conformal math via `vlabs-calibrate`. Customers stay in their own training loop and call us per batch — same pattern as Stripe's `Charge.create`. This eliminates the largest possible blast radius (RCE) by design.
2. **Single repo, separate service tree.** All new code lives under `services/api/` (FastAPI + SQLAlchemy + Stripe) and `services/landing/` (Next.js dashboard). Zero edits to `packages/vlabs-calibrate/`, `src/verifiable_labs_envs/`, or existing tests. The API depends on `vlabs-calibrate==0.1.0a1` like any external customer.
3. **Stack is boring on purpose.** FastAPI · SQLAlchemy 2.x async · Postgres on Supabase · Clerk for user auth · Stripe Checkout + webhooks · Fly.io for the API · Cloudflare for DNS. Each piece has a free tier, each piece is one Google search away from a working example.
4. **Cost to launch: ~$21/mo all-in.** $20 Fly machine + ~$0.71/mo amortised domain. Supabase / Clerk / Sentry / BetterStack stay free for the first ~10K MAU. At 10 paying Pro customers ($990 MRR) total burn rises to ~$75/mo (~$915 net margin) — unit economics work from day one.
5. **Stage A is the hard one.** Stages B and C are mostly integration glue around Stage A's data model. The five risks worth thinking about are listed in §8; the biggest is webhook security (Stripe signature + idempotency keys), and that's a known-good pattern.

Total estimate: **27–34 engineering days** spread over 5 calendar weeks. Stage A 8–10 days, Stage B 6–8 days, Stage C 8–12 days, with ~3–4 days of slack across the whole phase for undiscovered work.

---

## 2. Architecture

```
                          ┌────────────────────┐
                          │  Customer Python   │
                          │  (trainer, eval,   │
                          │   reward fn)       │
                          └─────────┬──────────┘
                                    │ HTTPS, X-Vlabs-Key header
                                    │ JSON body of triples
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │           api.verifiable-labs.com                 │
        │           ── Fly.io, FastAPI, uvicorn ──          │
        │                                                   │
        │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
        │  │  routes/ │→ │  auth.py │→ │ rate-limiter │    │
        │  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
        │       │             │               │            │
        │       ▼             ▼               ▼            │
        │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
        │  │ schemas. │  │   db.    │  │  calibration │    │
        │  │  py      │  │  py      │  │     .py      │    │
        │  │(pydantic)│  │(SA 2.x)  │  │ (vlabs_      │    │
        │  └────┬─────┘  └────┬─────┘  │  calibrate)  │    │
        │       │             │        └──────┬───────┘    │
        │       ▼             ▼               │            │
        │  ┌──────────┐  ┌──────────┐         │            │
        │  │ billing. │←→│ webhook. │         │            │
        │  │  py      │  │  py      │         │            │
        │  │ (Stripe) │  │ (Stripe) │         │            │
        │  └──────────┘  └──────────┘         │            │
        └───────────────────┬─────────────────┴────────────┘
                            │ asyncpg
                            ▼
                ┌────────────────────────────┐
                │    Supabase Postgres       │
                │  users, api_keys, runs,    │
                │  usage_counters, subs      │
                └────────────────────────────┘
                            ▲
                            │ Clerk JWT cookies
                            │ Stripe webhook
                            │
        ┌───────────────────┴───────────────────────────────┐
        │           app.verifiable-labs.com                 │
        │  ── Cloudflare Pages, Next.js, shadcn/ui ──       │
        │                                                   │
        │   Sign in (Clerk) → Dashboard → API keys +        │
        │   coverage charts + billing + Upgrade (Stripe)    │
        └───────────────────────────────────────────────────┘
```

**Key boundaries.**

* The customer's reward function never crosses the network. Only its outputs cross.
* The dashboard never receives an API key in plaintext after creation — only the SHA-256 hash + 8-char prefix is stored.
* Stripe webhooks land at `api.verifiable-labs.com/v1/billing/webhook` and write to the same DB.
* Clerk handles sign-in for the dashboard; the API itself uses **only** API keys (no JWT acceptance on the data plane).

---

## 3. API specification

All endpoints under `/v1/`. JSON in / JSON out. Errors follow RFC 7807 problem-details. Auth via `X-Vlabs-Key: vlk_<random>` header on every request except `/health`.

### 3.1 `POST /v1/calibrate`

Calibrate on a batch of pre-computed triples and persist a `calibration_id` for later use.

**Request body**

```json
{
  "alpha": 0.1,
  "nonconformity": "scaled_residual",
  "traces": [
    {"predicted_reward": 0.92, "reference_reward": 0.88, "uncertainty": 0.05},
    {"predicted_reward": 0.40, "reference_reward": 0.35, "uncertainty": 0.10}
  ],
  "metadata": {"experiment": "rlhf-v3", "model": "qwen-1.5b"}
}
```

`alpha` ∈ (0, 1). `nonconformity` ∈ `{"scaled_residual","abs_residual","binary"}` (other values reject). `traces` length ≥ 2, ≤ 10⁶. `metadata` optional opaque user dict ≤ 8 KiB.

**Response 200**

```json
{
  "calibration_id": "cal_01J9X9R4YZ8ZR8H7Y3M3K7G7V0",
  "alpha": 0.1,
  "nonconformity": "scaled_residual",
  "n_calibration": 2,
  "quantile": 0.91,
  "target_coverage": 0.9,
  "nonconformity_stats": {"mean": 0.07, "std": 0.03, "median": 0.05, "min": 0.02, "max": 0.21},
  "created_at": "2026-04-30T19:30:00Z"
}
```

**Errors**

| HTTP | Code | When |
|---|---|---|
| 400 | `invalid_alpha` | α ∉ (0,1) |
| 400 | `traces_too_few` | `len(traces) < 2` |
| 400 | `traces_too_many` | `len(traces) > 1_000_000` |
| 400 | `unknown_nonconformity` | not in registry |
| 400 | `missing_required_keys` | trace missing `reference_reward` (or `uncertainty` for scaled_residual) |
| 401 | `invalid_api_key` | header missing or unknown |
| 402 | `quota_exceeded` | monthly trace quota for tier exhausted |
| 429 | `rate_limited` | per-tier RPM exceeded |

### 3.2 `POST /v1/evaluate`

Run a held-out evaluation against an existing `calibration_id`.

**Request body**

```json
{
  "calibration_id": "cal_01J9X9R4YZ8ZR8H7Y3M3K7G7V0",
  "traces": [
    {"predicted_reward": 0.7, "reference_reward": 0.65, "uncertainty": 0.1}
  ],
  "tolerance": 0.05
}
```

**Response 200**

```json
{
  "calibration_id": "cal_…",
  "target_coverage": 0.9,
  "empirical_coverage": 0.91,
  "n": 1,
  "n_in_interval": 1,
  "interval_width_mean": 0.32,
  "interval_width_median": 0.32,
  "tolerance": 0.05,
  "passes": true,
  "nonconformity": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "median": 0.5}
}
```

**Errors**: `404 calibration_not_found`, plus the auth/quota set above.

### 3.3 `GET /v1/audit/{calibration_id}`

Retrieve metadata for a past calibration. Useful for compliance / reproducibility audits.

**Response 200**

```json
{
  "calibration_id": "cal_…",
  "created_at": "2026-04-30T19:30:00Z",
  "alpha": 0.1,
  "nonconformity": "scaled_residual",
  "n_calibration": 500,
  "quantile": 1.6717,
  "target_coverage": 0.9,
  "nonconformity_stats": {"…": "…"},
  "evaluations": [
    {"n": 2000, "empirical_coverage": 0.915, "passes": true, "ts": "2026-04-30T19:35:00Z"}
  ],
  "metadata": {"experiment": "rlhf-v3"}
}
```

### 3.4 `GET /v1/usage`

```json
{
  "tier": "free",
  "quota": {"traces_per_month": 10000, "rpm": 100},
  "current_period": {"start": "2026-04-01", "end": "2026-04-30"},
  "usage": {"traces": 4321, "calibrations": 12, "evaluations": 7},
  "remaining": {"traces": 5679}
}
```

### 3.5 Bonus — `POST /v1/predict` (open question, see §10 Q1)

> If accepted: given `calibration_id` + `(predicted_reward, uncertainty)` pair, return the conformal interval. This is the actual hot path in production (one call per inference) — without it, customers must compute the interval client-side from `quantile`, which is fine but wasteful for the high-volume workload.

### 3.6 Cross-cutting headers

* `X-Vlabs-Key` — required, value `vlk_<22 chars base32>`.
* `Idempotency-Key` — optional client UUID; deduplicates POSTs within 24 h.
* `X-Request-ID` — server returns this for every request; clients log + send back on bug reports.

---

## 4. Database schema (Supabase Postgres)

5 tables, all timestamped UTC, all UUID primary keys (ULID-encoded `id` strings for sort-friendly logs).

```sql
-- 4.1 users
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email CITEXT NOT NULL UNIQUE,
  name TEXT,
  clerk_user_id TEXT UNIQUE,                  -- nullable until Clerk wires up
  stripe_customer_id TEXT UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at TIMESTAMPTZ
);
CREATE INDEX users_clerk_idx ON users (clerk_user_id) WHERE clerk_user_id IS NOT NULL;

-- 4.2 api_keys (only the SHA-256 hash + 8-char prefix is stored)
CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  key_hash BYTEA NOT NULL UNIQUE,             -- sha256(plaintext)
  key_prefix CHAR(8) NOT NULL,                 -- first 8 chars, for UI display
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ
);
CREATE INDEX api_keys_user_idx ON api_keys (user_id);
CREATE INDEX api_keys_active_idx ON api_keys (key_hash) WHERE revoked_at IS NULL;

-- 4.3 calibration_runs (one row per /v1/calibrate call)
CREATE TABLE calibration_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  alpha DOUBLE PRECISION NOT NULL,
  nonconformity TEXT NOT NULL,
  n_calibration INT NOT NULL,
  quantile DOUBLE PRECISION NOT NULL,
  nonconformity_stats JSONB NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_bytes INT NOT NULL,
  request_traces INT NOT NULL                   -- = n_calibration but kept separate for billing
);
CREATE INDEX calibration_runs_owner_idx ON calibration_runs (api_key_id, created_at DESC);

-- 4.4 evaluations (one row per /v1/evaluate call)
CREATE TABLE evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  calibration_id UUID NOT NULL REFERENCES calibration_runs(id) ON DELETE CASCADE,
  api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  n INT NOT NULL,
  empirical_coverage DOUBLE PRECISION NOT NULL,
  target_coverage DOUBLE PRECISION NOT NULL,
  passes BOOLEAN NOT NULL,
  tolerance DOUBLE PRECISION NOT NULL,
  interval_width_mean DOUBLE PRECISION NOT NULL,
  nonconformity_stats JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_traces INT NOT NULL
);
CREATE INDEX evaluations_calib_idx ON evaluations (calibration_id, created_at DESC);

-- 4.5 usage_counters (one row per (api_key, month) — UPSERT in hot path)
CREATE TABLE usage_counters (
  api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  month DATE NOT NULL,                          -- always day = 1
  traces_count BIGINT NOT NULL DEFAULT 0,
  calibrations_count INT NOT NULL DEFAULT 0,
  evaluations_count INT NOT NULL DEFAULT 0,
  PRIMARY KEY (api_key_id, month)
);

-- 4.6 subscriptions
CREATE TABLE subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  stripe_subscription_id TEXT UNIQUE NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('pro','team')),
  status TEXT NOT NULL,                         -- active, past_due, canceled, …
  current_period_start TIMESTAMPTZ NOT NULL,
  current_period_end TIMESTAMPTZ NOT NULL,
  cancel_at_period_end BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX subscriptions_user_idx ON subscriptions (user_id);
```

Migrations via Alembic. Initial migration is the snapshot above; every later DDL change is its own revision.

**Hot-path SQL (per /v1/calibrate)**

```sql
INSERT INTO usage_counters (api_key_id, month, traces_count, calibrations_count)
VALUES ($1, date_trunc('month', now())::date, $2, 1)
ON CONFLICT (api_key_id, month) DO UPDATE
SET traces_count       = usage_counters.traces_count + EXCLUDED.traces_count,
    calibrations_count = usage_counters.calibrations_count + 1;
```

---

## 5. Stripe integration design

### 5.1 Products + prices (Stripe Dashboard, manual one-time setup)

| Product | Price | Type | Includes |
|---|---|---|---|
| `vlabs_pro` | $99 / mo | Recurring (subscription) | 1,000,000 traces / month |
| `vlabs_team` | $499 / mo | Recurring (subscription) | 10,000,000 traces / month |
| `vlabs_pro_overage` | $1.00 / 10K traces | Metered (usage_record) | beyond Pro quota |
| `vlabs_team_overage` | $0.40 / 10K traces | Metered (usage_record) | beyond Team quota |

Free tier has no Stripe object — encoded by *absence* of an `active` row in `subscriptions` for the user.

### 5.2 Checkout flow

1. Dashboard "Upgrade to Pro" → `POST /v1/billing/checkout` (auth: Clerk session, not API key).
2. Server creates a Stripe Checkout Session with line items = base price + metered price, `client_reference_id = users.id`. Returns the Checkout URL.
3. Browser redirected. User pays.
4. Webhook event `checkout.session.completed` arrives at `/v1/billing/webhook` → fetch session → upsert `subscriptions` row.
5. Subsequent monthly billing is automatic; overages are reported via `stripe.SubscriptionItem.create_usage_record`.

### 5.3 Webhook events handled (idempotent, signed)

| Event | Action |
|---|---|
| `checkout.session.completed` | Create / update `subscriptions` row, link `stripe_customer_id` to user |
| `customer.subscription.updated` | Update tier, status, period end |
| `customer.subscription.deleted` | Set status = canceled |
| `invoice.paid` | Confirm next period; reset overage counter |
| `invoice.payment_failed` | Flag user (banner in dashboard); no immediate downgrade |

Every webhook handler verifies `Stripe-Signature` header and stores the `event.id` in a `stripe_events` dedup table — replays are no-ops.

### 5.4 Metered billing logic (overage)

- Every `/v1/calibrate` and `/v1/evaluate` call increments `usage_counters.traces_count`.
- A daily cron task reads the diff vs. last reported value and posts `UsageRecord` to Stripe for each subscription whose tier has metered overage.
- Reset on `invoice.paid`.

For Stage A this is **stubbed**. Real metered billing lands in Stage B.

---

## 6. Deployment topology

| Surface | Host | Cost | DNS | TLS |
|---|---|---|---|---|
| `api.verifiable-labs.com` | Fly.io 1×shared-cpu, 1 GB | $20/mo | A → Fly | Fly-managed Let's Encrypt |
| `app.verifiable-labs.com` | Cloudflare Pages (Next.js SSG) | $0 | CNAME → Pages | Cloudflare-managed |
| `status.verifiable-labs.com` | BetterStack status page | $0 | CNAME → BetterStack | provider-managed |
| Postgres | Supabase free tier | $0 | n/a | provider-managed |
| Auth | Clerk free tier | $0 | n/a | provider-managed |

DNS held at Cloudflare Registrar (~$8.57/yr for `.com`). Cloudflare proxy on for `app.` and `status.` (default), proxy **off** for `api.` so we keep direct Fly hot path latency and Stripe-webhook signature stays clean.

Secrets live in Fly via `fly secrets set` and never in the repo:
- `DATABASE_URL` (Supabase pooler URL, transaction mode for serverless safety)
- `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID_PRO`, `STRIPE_PRICE_ID_TEAM`
- `CLERK_SECRET_KEY`, `CLERK_JWT_KEY`
- `SENTRY_DSN`
- `VLABS_API_KEY_HASH_PEPPER` (server-side pepper appended before SHA-256 of API keys)

For local dev: `.env.local` (gitignored), populated from a checked-in `.env.example` with placeholder values.

---

## 7. Stage-by-stage execution

### Stage A — Core API + auth + DB (Phase 16, weeks 1–2)

**Files created (≈ 25)**

```
services/api/
├── pyproject.toml                       # fastapi, uvicorn[standard], sqlalchemy[asyncio], asyncpg,
│                                         # alembic, slowapi, vlabs-calibrate==0.1.0a1, pydantic[email],
│                                         # python-ulid, structlog, pytest, httpx, [dev]
├── README.md                            # local dev setup; docker-compose for Postgres
├── PHASE_16_PLAN.md                     # this document, mirrored
├── SESSION_LOG.md                       # append-only progress log
├── .env.example                         # placeholders, gitignored .env.local copy
├── docker-compose.yml                   # local Postgres 16 for dev/tests
├── alembic.ini
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 0001_initial.py              # the §4 schema
├── src/vlabs_api/
│   ├── __init__.py                      # __version__ = "0.0.1"
│   ├── main.py                          # FastAPI(), /health, lifespan, middleware wiring
│   ├── config.py                        # pydantic-settings, reads env vars
│   ├── db.py                            # async engine, session factory, ORM models
│   ├── auth.py                          # X-Vlabs-Key middleware + DB lookup + rate limit hooks
│   ├── ratelimit.py                     # slowapi config keyed on api_key.id
│   ├── schemas.py                       # pydantic request/response models for all endpoints
│   ├── calibration.py                   # thin wrapper over vlabs_calibrate.calibrate / .evaluate
│   ├── usage.py                         # UPSERT helpers for usage_counters
│   ├── errors.py                        # RFC 7807 problem-details exception classes
│   └── routes/
│       ├── __init__.py
│       ├── calibrate.py
│       ├── evaluate.py
│       ├── audit.py
│       ├── usage.py
│       └── health.py
└── tests/
    ├── conftest.py                      # pgtest container fixture, fastapi TestClient
    ├── test_health.py
    ├── test_auth.py                     # missing/unknown/revoked keys
    ├── test_calibrate.py                # happy path, every error code in §3
    ├── test_evaluate.py
    ├── test_audit.py
    ├── test_usage.py
    └── test_ratelimit.py                # tier-aware bucket
```

**Milestones**

| # | Deliverable | Test |
|---|---|---|
| A1 | `pyproject.toml`, scaffold, `pip install -e .` works | import-only smoke |
| A2 | Postgres + Alembic + initial migration | `alembic upgrade head` succeeds against docker-compose Postgres |
| A3 | `auth.py` + `api_keys` CRUD via tests | `test_auth.py` green |
| A4 | `routes/calibrate.py` end-to-end | `test_calibrate.py` covers all §3.1 error codes |
| A5 | `routes/evaluate.py` and `routes/audit.py` | green tests, audit returns 404 cleanly |
| A6 | `routes/usage.py` reflects `usage_counters` UPSERTs | green |
| A7 | `ratelimit.py` enforces tier RPM | green |
| A8 | `pytest` from repo root still 519 + new (≥ 30 new) | hard gate |

**Gate at end of Stage A**: single commit `feat(api): Phase 16 Stage A — FastAPI core + auth + DB`, push to main, stop and wait for review. **Do not** start Stage B without explicit go.

### Stage B — Stripe + landing (week 3)

```
services/api/src/vlabs_api/
├── billing.py                           # Stripe Subscription / UsageRecord helpers
└── routes/
    ├── billing.py                       # POST /v1/billing/checkout (Clerk-auth)
    └── webhook.py                       # POST /v1/billing/webhook (signed)

services/landing/                        # Next.js 14 app router
├── package.json                         # next, react, @clerk/nextjs, shadcn/ui, swr
├── README.md
├── app/
│   ├── layout.tsx
│   ├── page.tsx                         # marketing landing
│   ├── (dashboard)/
│   │   ├── layout.tsx                   # Clerk SignedIn wrapper
│   │   ├── api-keys/page.tsx
│   │   ├── usage/page.tsx
│   │   └── billing/page.tsx             # "Upgrade to Pro / Team"
│   └── api/
│       └── checkout/route.ts            # proxies to /v1/billing/checkout with Clerk JWT
├── components/                          # shadcn/ui primitives + dashboards
└── lib/
    └── api.ts                           # typed client for /v1/*

services/api/tests/
├── test_billing_checkout.py             # mocks Stripe via stripe-mock
└── test_webhook.py                      # signed payload roundtrip
```

**Milestones**

| # | Deliverable |
|---|---|
| B1 | Stripe products created in Dashboard, IDs in `.env.example` placeholders |
| B2 | `routes/billing.py` returns Checkout URL given a Clerk-authed user |
| B3 | `routes/webhook.py` verifies signature, persists `subscriptions` rows, dedupes via `stripe_events` table |
| B4 | Next.js dashboard skeleton renders signed-in/signed-out states correctly |
| B5 | API key generation surface in dashboard creates a key, shows it once, then only the prefix |
| B6 | Usage page renders current month's counters from `/v1/usage` |
| B7 | Stripe end-to-end happy path tested with `stripe-mock` and Clerk dev keys |

**Gate**: commit `feat(api): Phase 16 Stage B — Stripe + landing page`, push, stop.

### Stage C — Production deploy + dashboards (weeks 4–5)

```
services/api/deploy/
├── Dockerfile                           # multi-stage, distroless final, non-root user
├── fly.toml                             # primary region: ams (close to EU customers)
├── fly.staging.toml                     # optional second app for staging
└── grafana_dashboards.json              # importable into BetterStack / Grafana Cloud free
```

**Milestones**

| # | Deliverable |
|---|---|
| C1 | Dockerfile builds on Apple Silicon and amd64 with `--platform=linux/amd64` |
| C2 | `fly launch --no-deploy` + secrets set + `fly deploy` → /health returns 200 |
| C3 | DNS records: `api.` (Fly), `app.` (Cloudflare Pages), `status.` (BetterStack) live with TLS |
| C4 | Sentry DSN wired; one synthetic error reaches the project |
| C5 | BetterStack uptime monitor pinging `/health` every minute, status page public |
| C6 | Coverage dashboard in landing app: per-API-key empirical-coverage line chart over time |
| C7 | Drift alert: cron task flags when most-recent `evaluate` run shows `|empirical − target| > 0.05`, emails user via Clerk |
| C8 | Smoke test: real customer flow (sign up → key → curl `/v1/calibrate` → see in dashboard → `Upgrade` → succeeds) — recorded once with `asciinema` for the demo |

**Gate**: commit `feat(api): Phase 16 Stage C — production deploy + dashboards`, push.
*Optional* CI workflow update (`.github/workflows/deploy.yml`) is gated on **explicit user approval** per Hard Constraint #1.

---

## 8. Risk register (top 5)

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | A user uploads a Python reward-fn string that we then `exec` somewhere — RCE | **Eliminated** | catastrophic | API design forbids it. The endpoint shape is `(predicted, reference, σ)` triples only. There is **no code path** that accepts user code. |
| **R2** | Stripe webhook spoofing or replay → fake "Pro" upgrade | low if we follow the playbook | high (revenue + trust) | (a) verify `Stripe-Signature` on every webhook, fail closed; (b) idempotency via `stripe_events.event_id UNIQUE`; (c) `Idempotency-Key` header on POSTs from our side. |
| **R3** | Rate limit bypass under load (single-instance slowapi vs. multi-instance scale-up) | medium once we scale | medium (DDOS, quota abuse) | Stage A ships single-instance + slowapi (sufficient for ≤ a few hundred RPS). Stage C upgrade path: drop in Redis-backed limiter (`slowapi[redis]`), no API change. |
| **R4** | Postgres hot-path UPSERT on `usage_counters` becomes a write-throughput bottleneck | low at < 100 RPS, real at > 1k RPS | medium | (a) batch increments per-request with a single UPSERT (already in design); (b) when traffic warrants, add Redis counters with periodic flush to Postgres. |
| **R5** | Free-tier abuse: signups with throwaway emails creating endless 10K-trace quotas | medium once we're discoverable | low–medium ($ small, but signal noise) | (a) rate-limit signup per IP via Cloudflare; (b) require email verification (Clerk default); (c) cap **per-IP** monthly traces independently of per-key quota. |

Other risks tracked but below the top-5 line: Supabase free-tier disk limits (500 MB ≈ ~10 M `calibration_runs` rows), Fly cold-start latency on free shared-cpu (mitigated by the $20 plan), GDPR data-deletion request handling (manual playbook in `services/api/docs/dpa.md`).

---

## 9. Cost estimate

### 9.1 Pre-revenue (today → first paying customer)

| Item | $/mo | Notes |
|---|---|---|
| Domain `verifiable-labs.com` | $0.71 | Cloudflare Registrar at-cost, $8.57/yr ÷ 12 |
| Fly.io API (1 GB shared-cpu) | $20.00 | Could start at $5 (256 MB) — but 256 MB is tight once we load `vlabs_calibrate` + asyncpg connections; 1 GB gives headroom. |
| Supabase free tier | 0.00 | 500 MB DB, 2 GB egress |
| Cloudflare Pages | 0.00 | Unlimited static, 500 builds/month |
| Clerk | 0.00 | First 10K MAU free |
| Sentry | 0.00 | 5K events/mo free |
| BetterStack | 0.00 | 1 monitor + 1 status page free |
| **Total** | **$20.71** | rounded to ~$21/mo |

### 9.2 At 10 paying Pro customers (MRR $990)

| Item | $/mo | Notes |
|---|---|---|
| Pre-revenue baseline | 20.71 | unchanged |
| Stripe processing | ~30.00 | 2.9% + $0.30 × 10 transactions |
| Supabase Pro | 25.00 | upgrade for SLA + bigger DB once we have paying users (not strictly required at 10 customers; padding) |
| **Total** | **~$75.71** | ~$915 net margin |

### 9.3 Break-even

* Per Pro customer: $99 − $0.30 − $99 × 2.9% ≈ **$95.83 net** after Stripe.
* Fixed costs: ~$25 baseline (Fly + domain + Supabase Pro buffer).
* **Break-even at ~1 paying customer.** Anything beyond is upside.

### 9.4 Worst-case overrun (Pro at 1.5× quota every month)

* 1.5 M traces × $0.10 / 10K = $15 overage / customer / mo, billed automatically — neutral.

---

## 10. Open questions for Stelios

1. **`/v1/predict` endpoint** — your spec listed exactly 4 endpoints. The hot path in production is "given calibrated quantile, return interval per inference," and without `/v1/predict` customers either compute it client-side from the returned `quantile` or call `/v1/evaluate` with `n=1`. Add `/v1/predict` to v0.1 (recommended), or hold to 4 endpoints?
2. **User auth** — Clerk (recommended; ~30 min wire-up, free ≤ 10K MAU) or roll-your-own JWT? Clerk also ships sign-in UI we don't have to build.
3. **Database** — Supabase (auth + storage + Postgres in one place, free 500 MB) or Neon (cleaner Postgres, free 3 GB, no auth bundling)? Recommend Supabase for the bundling; Neon is the better long-term DB choice.
4. **Stripe entity** — billing entity for the company. Greek personal · EE Greek LLC · Estonian e-Residency OÜ · US LLC? Each has different fee + tax + paperwork profile. Out of scope for the code, but I won't push the Stripe products live until you tell me which entity.
5. **What is "1 trace"?** For billing: is `n_calibration` the unit, or does each `evaluate` row also count as a trace, or only `calibrate` rows? Recommendation: bill on `n_calibration + n_evaluate_traces` summed across calls in a month — the customer pays for every triple they push through us.
6. **Free-tier abuse posture** — strict (require credit card on file, verified email, 1 free per company) or loose (frictionless 30-second signup)? Strict is safer for billing fraud, loose is better for the funnel. Recommend loose for the first 90 days of the launch funnel + monitoring; tighten if abuse appears.
7. **Default region for Fly.io** — `ams` (Amsterdam, lowest latency for EU customers including Stelios on Colab) or `iad` (US east, closer to most ML labs)? You can multi-region later.
8. **`vlabs-calibrate` version pinning policy** — pin to exact `0.1.0a1` (Stage A) and bump in lockstep with SDK releases, or use `>=0.1.0a1,<0.2`? Recommend exact pin; SaaS reproducibility outweighs convenience.

---

## 11. What this plan does **not** propose (intentional)

* **No edits to** `packages/vlabs-calibrate/`, `src/verifiable_labs_envs/`, `examples/training/`, `tests/`, or `.github/workflows/`. Hard Constraint #1 is honoured.
* **No paid-tier feature gating beyond rate + quota** in v0.1. Tiers differ by RPM and trace quota; the *features* are identical so the upgrade prompt has a clean CTA ("you hit your quota — upgrade").
* **No multi-tenant / org accounts** in v0.1. One user = one billing entity. Org accounts land in Phase 17 if customer feedback demands it.
* **No SSO / SAML** in v0.1. Clerk's email + Google + GitHub login is enough.
* **No SDK changes**. The SaaS is a thin client of `vlabs-calibrate==0.1.0a1`. SDK improvements continue independently per the existing release schedule.
* **No CI auto-deploy in this phase**. Stage C ships `fly deploy` from the dev's laptop. CI auto-deploy is one isolated `.github/workflows/deploy.yml` patch *after* Stage C, gated on your explicit approval per Hard Constraint #1.

---

## 12. Approval gate

After you read §1–§10 and answer the §10 questions (or wave them off), I will start **Stage A only**. I will:

1. Stage exclusively under `services/api/`.
2. Run repo-root `pytest` after each milestone — 519 tests must stay green.
3. Run secret pattern scan before every commit.
4. Stop at the end of Stage A and wait for your review before B.

When you say go, I'll work through A1 → A8, then commit `feat(api): Phase 16 Stage A — FastAPI core + auth + DB`, push to main, and report.
