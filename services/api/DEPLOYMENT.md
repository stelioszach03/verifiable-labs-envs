# vlabs-api — Deployment

> Phase 16 Stage C playbook. Stripe is **deferred** (kill-switch
> `VLABS_BILLING_ENABLED=false`); flip to `true` once the Delaware C-corp
> registration completes.

## Architecture

```
                            api.verifiable-labs.com
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Fly.io  app=vlabs-api  │
                          │  region=iad             │
                          │  vm=shared-cpu-1x 1GB   │
                          │  scale=1..3 machines    │
                          └────┬───────────┬────────┘
                               │           │
                       ┌───────┘           └────────┐
                       ▼                            ▼
              ┌───────────────┐         ┌───────────────────┐
              │ Supabase pg17 │         │ Upstash Redis REST │
              │ (us-west-1)   │         │ (rate limit ZSET)  │
              └───────────────┘         └───────────────────┘
```

## First-time deploy (one-time)

Pre-reqs: `flyctl auth login`, `services/api/.env.local` populated, audit green.

```bash
cd services/api
./deploy/first-deploy.sh
```

The script:
1. `flyctl launch --no-deploy --copy-config --name vlabs-api --region iad`
2. Parses `.env.local` (no `source` — special chars safe), pushes every value via
   `flyctl secrets set --stage`
3. `flyctl deploy --remote-only`
4. `flyctl certs create vlabs-api api.verifiable-labs.com`
5. Sanity checks `/health`

Expected duration: 4–6 minutes (mostly Docker build + image upload).

## DNS

Cloudflare zone for `verifiable-labs.com`:

| Record | Value | Proxy |
|---|---|---|
| A `api` | `<Fly.io v4 IP from flyctl ips list>` | DNS-only (orange cloud OFF) |
| AAAA `api` | `<Fly.io v6 IP>` | DNS-only |
| CNAME `app` | `vlabs-landing.pages.dev` | Proxied (orange cloud ON) |

`api.` proxy must stay OFF — Stripe webhook signature verification needs the
direct request body, and Cloudflare's proxy can rewrite headers.

## Subsequent deploys

```bash
cd services/api
flyctl deploy --remote-only
```

That's it. The Dockerfile build context is `services/api/`; migrations apply
on container boot via `entrypoint.sh`.

## Rollback

```bash
# List recent releases
flyctl releases --app vlabs-api

# Roll back to a specific version
flyctl deploy --image registry.fly.io/vlabs-api:v<N> --app vlabs-api

# Or use the previous image tag
flyctl machine restart --app vlabs-api
```

Database migrations are NOT auto-rolled-back. If a migration broke production:

1. `flyctl ssh console --app vlabs-api`
2. `alembic downgrade -1`
3. Redeploy a known-good image

Better path for breaking changes: ship them in two phases (additive change first,
remove-old-column-second deploy after the new code is stable).

## Secrets rotation

Any secret in `flyctl secrets list` can be rotated without redeploy:

```bash
flyctl secrets set --app vlabs-api STRIPE_SECRET_KEY="sk_test_new..."
# Fly automatically restarts machines to pick up the new value.
```

For the DB password specifically: rotate in Supabase dashboard first, then update the secret. The connection pool reconnects on next request.

For `VLABS_API_KEY_HASH_PEPPER` rotation: this invalidates **all existing user
API keys** (their hashes become unmatchable). Don't rotate without a plan to
re-issue every active key. If forced to rotate (e.g., leak), the playbook is:

1. Send users an email warning (manual, no automation yet)
2. `flyctl secrets set VLABS_API_KEY_HASH_PEPPER=...`
3. Watch Sentry for `invalid_api_key` 401 spikes
4. Users regenerate keys via the dashboard

## Debugging a sick deploy

```bash
flyctl logs --app vlabs-api               # tail logs
flyctl ssh console --app vlabs-api        # interactive shell on a machine
flyctl status --app vlabs-api             # machine state
flyctl checks list --app vlabs-api        # health-check history
```

Sentry catches unhandled exceptions automatically — check the project
configured in `SENTRY_DSN`.

## Cost (steady state)

| Item | $/mo |
|---|---|
| Fly.io shared-cpu-1x 1GB × 1 (always on) | 5 |
| Fly.io scale-out (each additional machine) | 5 |
| Supabase Pro (only if traffic exceeds free tier) | 25 |
| Upstash Redis REST (free tier covers 10K req/day) | 0 |
| Cloudflare Pages (landing) | 0 |
| Sentry (5K events/mo free) | 0 |
| BetterStack (1 monitor + 1 status page free) | 0 |
| Domain `verifiable-labs.com` | 0.71 |
| **Pre-revenue total** | **~$6/mo** |

At 10 paying Pro customers ($990 MRR): add ~$30/mo Stripe fees + Supabase Pro
upgrade ($25) → **~$61/mo total**, ~$929/mo net margin.

## Going-live checklist

- [ ] `services/api/.env.local` populated, all 14 vars present (audit green).
- [ ] `flyctl auth whoami` returns the right account.
- [ ] DNS records created in Cloudflare for `api.` and `app.`.
- [ ] `./deploy/first-deploy.sh` completes without errors.
- [ ] `curl https://api.verifiable-labs.com/health` returns 200.
- [ ] Sentry receives a synthetic test event.
- [ ] BetterStack monitor pinging `/health` every minute.
- [ ] Cloudflare Pages app deployed (`services/landing/deploy/cloudflare-deploy.sh`).
- [ ] Sign in at `https://app.verifiable-labs.com`, mint an API key, smoke-test
      with `/v1/calibrate`.
- [ ] Stripe **stays deferred** (`VLABS_BILLING_ENABLED=false`) until
      Delaware C-corp lands.

See [`RUNBOOK.md`](./RUNBOOK.md) for incident response procedures.
