# vlabs-api — Hosted Calibration API

The paid SaaS layer wrapping `vlabs-calibrate==0.1.0a1`. Stage A of Phase 16:
core API + auth + DB. Stage B (Stripe + landing) and Stage C (deploy +
dashboards) are separate sessions.

> **Status: 0.0.1 alpha.** Code-complete for Stage A — core endpoints
> work end-to-end against a local Postgres. Not yet deployed.

## Local development

The repo at `services/api/` is a standalone Python package. Two ways to
run a development Postgres:

### Option A — `pgserver` (zero install)

```bash
cd services/api
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests — pgserver downloads its own Postgres binary on first use
pytest
```

`pgserver` is the recommended path on Linux, macOS, and Windows where
Docker isn't running. It bundles a self-contained Postgres binary.

### Option B — Docker Compose (matches production)

```bash
cd services/api
docker compose up -d postgres
export DATABASE_URL="postgresql+asyncpg://vlabs:vlabs@localhost:5433/vlabs"
alembic upgrade head
pip install -e ".[dev]"
uvicorn vlabs_api.main:app --reload
```

API at <http://localhost:8000>, OpenAPI at <http://localhost:8000/docs>.

## Configuration

All config is via environment variables, validated by `pydantic-settings`.
Copy `.env.example` to `.env.local` and fill in:

```bash
cp .env.example .env.local
# edit .env.local
```

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes (prod) | `postgresql+asyncpg://user:pass@host:port/db` |
| `VLABS_API_KEY_HASH_PEPPER` | yes (prod) | server-side secret appended before SHA-256 of plaintext keys |
| `VLABS_LOG_LEVEL` | no | default `INFO` |
| `VLABS_ENVIRONMENT` | no | `dev` / `staging` / `prod`; gates a few behaviours |
| `STRIPE_SECRET_KEY` | Stage B | Stripe server key (test mode until C-corp registered) |
| `STRIPE_WEBHOOK_SECRET` | Stage B | endpoint signing secret |
| `CLERK_SECRET_KEY` | Stage B | Clerk backend key for dashboard JWT verification |

For Stage A only `DATABASE_URL` and `VLABS_API_KEY_HASH_PEPPER` are required.

## Endpoints — what's live in Stage A

All under `/v1/`. Auth via `X-Vlabs-Key` header (except `/health`).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, no auth |
| `POST` | `/v1/calibrate` | calibrate from triples → `calibration_id` + quantile |
| `POST` | `/v1/predict` | given a `calibration_id` + `(predicted, sigma)`, return interval |
| `POST` | `/v1/evaluate` | held-out coverage report against a `calibration_id` |
| `GET`  | `/v1/audit/{id}` | retrieve calibration + evaluation history |
| `GET`  | `/v1/usage` | current month's traces vs tier quota |

Full spec in [`PHASE_16_PLAN.md`](./PHASE_16_PLAN.md) §3.

## Database

Schema lives in `migrations/versions/0001_initial.py`. Six tables on
Postgres: `users`, `api_keys`, `calibration_runs`, `evaluations`,
`usage_counters`, `subscriptions`. See `PHASE_16_PLAN.md` §4 for the
full DDL and indexes.

To apply against any Postgres:

```bash
DATABASE_URL=postgresql+asyncpg://... alembic upgrade head
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The full suite runs in ~2–4 s once `pgserver` has cached its binary.
Tests are NOT yet wired into the repo-root `pytest` (Phase 16 Stage C
will add `services/api/tests/` to root `testpaths` + the CI workflow).

## Production deploy

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full Stage C playbook
(Fly.io app, secrets, DNS, certs, rollback). Tldr:

```bash
cd services/api
./deploy/first-deploy.sh         # one-time
flyctl deploy --remote-only      # subsequent deploys
```

[`RUNBOOK.md`](./RUNBOOK.md) covers incident response — DB outage,
rate-limit spikes, OOM, SSL renewal, webhook errors.

> **Stripe is deferred** in Stage C: `VLABS_BILLING_ENABLED=false`
> ships in production. `/v1/billing/*` routes return `503 billing_not_activated`,
> the webhook short-circuits. Flip to `true` after the Delaware C-corp
> registration via Stripe Atlas completes.

## License

Apache-2.0 — see repo root [LICENSE](../../LICENSE).
