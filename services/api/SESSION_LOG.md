# Session log — vlabs-api

Append-only timestamped log. Newest entry on top.

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
