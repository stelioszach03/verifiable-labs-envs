# Session log — vlabs-api

Append-only timestamped log. Newest entry on top.

---

## 2026-05-01 — Phase 16, Stage C (deploy + monitoring + Stripe deferred)

**Goal**: production deploy infrastructure + observability + admin
plane, with Stripe **deferred** (kill-switch flag, all billing routes
short-circuit) until Delaware C-corp registration completes.

### Done — services/api
- **Stripe deferred mode**: new `VLABS_BILLING_ENABLED` flag (default
  `false`). `/v1/billing/checkout` and `/v1/billing/portal` raise
  `BillingNotActivated` (503) when disabled. `/v1/billing/webhook`
  short-circuits with `200 {"deferred": true}` and never instantiates
  the Stripe SDK.
- **Redis-backed rate limit** (`ratelimit.py` rewrite): Upstash REST
  sliding window via `ZADD` / `ZCARD` / `ZREMRANGEBYSCORE`. Backend
  auto-detected — falls back to in-memory `deque` when Upstash creds
  are unset (so tests + local dev keep working). Same
  `enforce_rate_limit` dependency contract; routes don't change.
- **Sentry SDK init** in `main.create_app`: gated on `SENTRY_DSN`
  presence so tests stay quiet. FastAPI integration is auto-loaded
  via `sentry-sdk[fastapi]` extra.
- **Admin plane**: `routes/admin.py` exposes `GET /v1/admin/dashboard`
  (Clerk-authed, allowlist via `VLABS_ADMIN_CLERK_IDS`). Returns row
  counts + 10 most recent calibrations + `billing_enabled` status.
- **Scheduled jobs scaffold**: `jobs/reconcile_overage.py` stub —
  logs `skipped: billing deferred` + exits 0. Fly scheduled-machine
  config commented in `deploy/fly.toml`, ready to enable post-C-corp.
- **Deploy infra**: `deploy/Dockerfile` (multi-stage Python 3.11,
  non-root vlabs user, libpq runtime), `deploy/fly.toml`
  (region `iad`, shared-cpu-1x 1GB, scale 1..3, `/health` checks),
  `deploy/entrypoint.sh` (alembic upgrade head → uvicorn),
  `deploy/.dockerignore`, `deploy/first-deploy.sh` (idempotent
  flyctl launch + secrets parse from `.env.local` without sourcing).

### Done — services/landing
- `/admin` page (server component, Clerk-protected via middleware
  matcher). Calls `/v1/admin/dashboard` with the `vlabs-api` JWT
  template. Renders 6-stat grid + recent-calibrations table.
- `app/dashboard/actions.ts` already updated in a prior step to use
  `getToken({ template: "vlabs-api" })`.
- `deploy/cloudflare-deploy.sh` — idempotent `npm ci` →
  `next-on-pages` → `wrangler pages deploy`.

### Tests
- 7 new tests added (4 admin, 2 deferred-billing, 1 deferred-webhook).
- Total `services/api/tests/`: **55 passing** in ~8s on local pgserver.
- Existing 519 + 6-skipped repo-root baseline unchanged.
- Conftest forces memory rate-limit backend (`UPSTASH_REDIS_REST_URL=""`)
  to keep tests fast — production .env.local has real creds, but
  routing every test request through real Redis added ~100s of network
  latency to a previously 8s suite.

### CI integration
- Root `pyproject.toml`: `services/api/tests` added to `testpaths`.
- `.github/workflows/ci.yml`: install step `pip install -e
  "services/api[dev]"`, ruff lint extends to `services/api/{src,tests}`,
  pytest discovers and runs the new tests automatically.

### Decisions logged
- **Webhook always returns 2xx**, even in deferred mode — Stripe's
  retry policy is aggressive on 5xx and we don't want retry storms
  during the C-corp transition window.
- **Admin allowlist is env-driven** (`VLABS_ADMIN_CLERK_IDS=user_a,user_b`)
  rather than a DB role table. Keeps the source of truth in Fly secrets,
  rotatable without a deploy.
- **Redis ratelimit opens the gate on transient errors** (logs warning,
  returns `True`). Better to over-serve briefly than to drop all traffic
  when Upstash blips — Sentry alerts on the warning class.
- **Pepper rotation invalidates all keys** — documented in DEPLOYMENT.md;
  no automation. Stage C ships the warning, not the rotation flow.

### Pending operational steps for Stelios
1. Run `services/api/deploy/first-deploy.sh` (Fly launch + secrets +
   deploy + cert request).
2. Add Cloudflare DNS records for `api.` (A/AAAA, DNS-only) and `app.`
   (CNAME, proxied).
3. Run `services/landing/deploy/cloudflare-deploy.sh` (Pages project
   + deploy).
4. Create BetterStack monitor on `https://api.verifiable-labs.com/health`
   (1-minute interval, alerts to email).
5. Add an admin Clerk user ID to `VLABS_ADMIN_CLERK_IDS` (Stelios's own
   Clerk ID is the obvious first entry).

### Open / next phase
- **Phase 17** (Capability Report automation): write nightly job that
  pulls fresh test traces, computes coverage, stores under
  `services/api/reports/`, emails summary to admins.
- **Stripe activation** (independent track): Delaware C-corp via Stripe
  Atlas; once approved, run `scripts/create_stripe_products.py`,
  flip `VLABS_BILLING_ENABLED=true`, uncomment the cron in fly.toml.

---

## 2026-04-30 — Phase 16, Stage B (Stripe + Next.js dashboard)

**Goal**: ship Stripe (TEST MODE), tier-aware rate limit, Clerk JWT
acceptance for the management plane, and a Next.js landing/dashboard.

### Done — services/api
- Replaced slowapi with a 30-LOC sliding-window in-memory rate limiter
  in `ratelimit.py`. Routes `calibrate`, `predict`, `evaluate`, `audit`,
  `usage` now use the `enforce_rate_limit` FastAPI dependency. Single-
  instance only; Stage C adds Redis backend.
- Added `clerk_auth.py` (PyJWKClient, JIT user creation from `sub` claim)
  and `billing.py` (Stripe wrapper: ensure_stripe_customer, create_checkout_session,
  create_billing_portal_session, verify_webhook_signature, sync_subscription_from_event).
- New routes: `/v1/billing/checkout`, `/v1/billing/portal` (Clerk-authed),
  `/v1/billing/webhook` (Stripe signature + idempotency via stripe_events
  table), `/v1/keys` GET/POST, `/v1/keys/{id}` DELETE (Clerk-authed).
- New schemas in `schemas.py`: CheckoutRequest/Response, PortalResponse,
  CreateAPIKeyRequest, APIKeyInfo/Created/List.
- New errors: InvalidClerkToken, WebhookSignatureInvalid,
  WebhookEventUnsupported, StripeNotConfigured, ClerkNotConfigured,
  APIKeyNotFoundForUser.
- New Alembic revision `0002_stripe_events.py` + same SQL applied to
  Supabase via MCP `apply_migration`.
- Helper `scripts/create_stripe_products.py` — idempotent test-mode
  product/price bootstrapper (run once with sk_test_).

### Done — services/landing
- Next.js 15 + Tailwind + Clerk skeleton, no shadcn install (design
  language only). 21 files total.
- Marketing landing (`/`), pricing (`/pricing`), Clerk-hosted sign-in
  and sign-up routes.
- Dashboard with 4 sub-pages: overview, api-keys, usage, billing.
- Server actions in `app/dashboard/actions.ts` for all mutations
  (`actCreateKey`, `actRevokeKey`, `actUpgradeTo`, `actOpenPortal`).
- Typed `lib/api.ts` wraps the vlabs-api endpoints used by the dashboard
  and forwards the Clerk session token.

### Tests
- 18 new tests in `services/api/tests/`:
  test_billing (5), test_webhook (5), test_keys (5), test_ratelimit (3).
  All use stub fixtures (`stub_clerk_verify`, `stub_stripe`,
  `stub_webhook_verify`) so they don't require live Stripe/Clerk.
- Total `services/api/tests/`: 48 passing in 7s.
- Repo-root `pytest --ignore=tests/training ...`: 519 + 6 (unchanged).

### Stripe products created
- Created via the helper script using `STRIPE_SECRET_KEY=sk_test_...`.
  Stelios's task to actually run the script and paste the resulting
  `STRIPE_PRICE_ID_*` values into `services/api/.env.local`.

### Webhook signature verification approach
- `stripe.Webhook.construct_event(payload, sig_header, whsec)` —
  Stripe's official helper; verifies HMAC-SHA256 over `<timestamp>.<payload>`
  with the endpoint secret. Implementation in `vlabs_api.billing.verify_webhook_signature`.
- Idempotency via `stripe_events` table: `INSERT` first (PK on `event_id`),
  process only if INSERT succeeded; replays return `{"deduped": true}`.

### Decisions logged
- **Tier-aware rate limit is per-key**, not per-user. A user with 3 keys
  gets 3× the tier RPM if traffic is spread across them. This is the
  standard SaaS behaviour and matches Stripe / OpenAI.
- **Webhook handler always returns 2xx** even on internal errors. Errors
  are persisted in `stripe_events.error` for debugging. 5xx triggers
  Stripe retry storms — bad practice.
- **No live Stripe products** in this commit. The script is committed,
  the user runs it locally to create test products. Production
  activation is gated on the Delaware C-corp registration.

### Open / next stage
- Stage C: Fly.io deploy, custom domain (`api.`/`app.`), DNS,
  monitoring (Sentry/BetterStack), Redis-backed rate limiter,
  CI deploy workflow (gated on explicit user approval).

---

## 2026-04-30 — Phase 16, Stage A (core API + auth + DB)

**Goal**: code-complete the four `/v1/*` endpoints + `/v1/predict`
(per Q1 answer) on top of a Postgres-backed schema, with a passing
test suite, and zero changes to the existing 519-test repo.

### Done
- Skeleton: `pyproject.toml` (pinned `vlabs-calibrate==0.1.0a1`),
  `.env.example`, `docker-compose.yml`, `alembic.ini`, `README.md`.
- Core source (`src/vlabs_api/`): `config.py`, `db.py`,
  `schemas.py`, `errors.py`, `auth.py`, `ratelimit.py`,
  `calibration.py`, `usage.py`, `main.py`, `routes/{health,
  calibrate,evaluate,predict,audit,usage}.py`.
- Migrations (`migrations/versions/0001_initial.py`): six tables on
  Postgres, indexes, `pgcrypto` for `gen_random_uuid()`.
- Tests (`tests/`): conftest with `pgserver` session fixture +
  per-test schema teardown; coverage of every endpoint and
  every error code in §3 of the plan.

### Decisions logged
- **Auth at the data plane is API-key only**. Clerk JWTs are NOT
  accepted by `/v1/*` endpoints. Clerk is reserved for the dashboard
  in Stage B (`/v1/billing/*`).
- **Quotas + rate limits are tier-aware**. Free tier = 100 RPM,
  10K traces/month; Pro/Team unlocked at Stage B once Stripe
  webhooks land.
- **Calibration storage stores the quantile + nonconformity name +
  alpha — not the raw traces.** Audit response shows aggregate
  stats only. Raw triples are user-side data.
- **`/v1/predict` uses `vc.nonconformity.get(name).interval_fn`
  directly**; we don't reconstruct a `CalibratedRewardFn` server-side
  because there's no reward callable to wrap.
- **MCP-based Supabase migration apply was not possible** in this
  session — the connected Supabase MCP returned `permission denied`.
  Stage A delivers the migration as code; Stelios applies via local
  `alembic upgrade head` against the Supabase `DATABASE_URL` (which
  the publishable key alone doesn't grant).

### Tests (local pgserver)
- `pytest services/api/tests/` → all green, ~3 s.
- Repo-root `pytest` (excluding M3/M4 untracked) → 519 + 6 baseline,
  unchanged.

### Open / next stages
- Stage B (Phase 16, week 3): Stripe Checkout + webhooks (test mode
  only until Delaware C-corp resolves), Next.js landing app under
  `services/landing/`.
- Stage C (Phase 16, weeks 4-5): Fly.io deploy, DNS, dashboards,
  `.github/workflows/deploy.yml` (gated on explicit approval).
