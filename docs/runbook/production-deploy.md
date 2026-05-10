# Production deploy runbook

End-to-end procedure for deploying Phase 22-31 to Fly.io. Three-phase
orchestration with strict separation: CC owns discovery + scripts +
deploy execution; Stelios owns the manual silent-prompt secrets entry
in the middle.

**Time budget**: ~2 h end-to-end, ~15-30 min of which is Stelios at
the terminal typing missing secrets.

## Quickstart

```bash
# 1. CC has already produced the gap report + scripts.
cat reports/deploy/SECRETS_GAP_REPORT.md

# 2. (Stelios) generate the Fernet key for the R2 LLM-key encryption
#    field. Copy the output to the clipboard.
python scripts/deploy/generate_fernet_key.py

# 3. (Stelios) run the silent-prompt provisioner. It prompts only for
#    keys that are STILL MISSING from services/api/.env.local.
bash scripts/deploy/provision_production_secrets.sh

# 4. (Stelios) verify both .env and Fly.io agree.
bash scripts/deploy/verify_deploy_readiness.sh
# Expect: VERDICT: GO

# 5. Reply "secrets ready" to CC; CC executes Phase 3 (deploy).
```

## Phase 1 — Discovery (CC, automated)

CC inspects:

- `services/api/fly.toml` (env + Dockerfile reference).
- `services/api/src/vlabs_api/config.py` (Pydantic Settings — canonical
  recognised env vars).
- All other `os.environ.get(...)` call sites in `vlabs_api/**`.
- `services/api/.env.example` for default-fallback documentation.
- `services/api/.env.local` — keys only; values never read.

Output: `reports/deploy/SECRETS_GAP_REPORT.md`. The report enumerates
every required + optional production secret, marks which are already
in `.env.local`, and lists what's missing.

This phase is fully automated — no Stelios action required.

## Phase 2 — Provisioning (Stelios manual, ~15-30 min)

### Step 2a — Collect external dependencies

Stelios needs each of the following accounts/credentials before
running the provisioner. Most should already be set up from prior
phases.

| Credential                   | Where                                       | Cost (Tier 1)      |
|------------------------------|---------------------------------------------|--------------------|
| Supabase Postgres            | https://supabase.com (existing project)     | Free (500 MB)      |
| Upstash Redis                | https://upstash.com                         | Free (10K req/d)   |
| Cloudflare R2                | https://dash.cloudflare.com/<acct>/r2       | $0.015/GB-month    |
| Resend                       | https://resend.com                          | Free (3K msg/mo)   |
| Clerk                        | https://dashboard.clerk.com (existing)      | Free (10K MAU)     |
| Sentry                       | https://sentry.io (optional)                | Free (5K events)   |
| BetterStack                  | https://betterstack.com (optional)          | Free tier          |

### Step 2b — Generate the Fernet key

```bash
python scripts/deploy/generate_fernet_key.py
# Outputs a single line: 44-char base64 like:
# aGVsbG8td29ybGQtdGhpcy1pcy1ub3QtYS1yZWFsLWtleS1qdXN0LWFuLWV4YW1wbGU=
```

Copy that line. The provisioner will prompt for
`VLABS_DATA_LLM_KEY_ENCRYPTION` — paste it there.

**Critical**: the key is single-use. Once production data is
encrypted with it, rotating it in-place would invalidate every
existing payload. Future rotation is gated on Phase 31.G's key-id
versioning column.

### Step 2c — Run the silent-prompt provisioner

```bash
bash scripts/deploy/provision_production_secrets.sh
```

What it does:

1. Verifies `services/api/.env.local` is gitignored (refuses to write
   if not).
2. For each REQUIRED secret missing from the file, displays a
   one-line description + format hint, then `read -srp`s the value
   (input HIDDEN at terminal).
3. Writes the value to `services/api/.env.local` (chmod 600).
4. If `flyctl` is on PATH and the user is logged in, also stages the
   secret on Fly.io via `flyctl secrets set --stage`.
5. Repeats for OPTIONAL secrets (Sentry, BetterStack, Cloudflare,
   admin allowlist, Slack webhook).
6. Writes the production flag invariants explicitly:
   - `VLABS_ENVIRONMENT=prod`
   - `VLABS_LOCAL_FAKE_R2=false`
   - `VLABS_LOCAL_FAKE_EMAIL=false`
   - `VLABS_LOCAL_FAKE_PKI=false`
   - `VLABS_LOCAL_FAKE_HF=false`

Skip a prompt by pressing Enter on an empty line. Required secrets
left empty will fail the verification step in 2d.

The provisioner is **safe to re-run**: keys already provisioned in
`.env.local` are skipped. Only blank or missing entries are
re-prompted.

### Step 2d — Verify readiness

```bash
bash scripts/deploy/verify_deploy_readiness.sh
```

Output is a table of every required + optional secret, with two
columns: `local` (present in `.env.local`) and `fly` (present in
`flyctl secrets list`). The bottom shows the production flag
invariants and a final `VERDICT: GO` or `VERDICT: NO-GO`.

If `flyctl` is unavailable, the script reports Fly.io coverage as
"unavailable" and bases the verdict on `.env.local` + invariants
alone. **Re-run on a flyctl-equipped host before triggering the
deploy** so the Fly.io column is filled.

If verdict is NO-GO: re-run the provisioner.

### Step 2e — Hand off to CC

When the verifier says `VERDICT: GO`, reply to CC with the literal
string **"secrets ready"**. CC then proceeds to Phase 3 automatically.

## Phase 3 — Deploy (CC, automated, ~30 min)

Triggered only after Stelios's "secrets ready" message.

### Step 3a — Pre-deploy checks

```bash
bash scripts/deploy/verify_deploy_readiness.sh
bash scripts/preflight/check_deploy_readiness.sh
python scripts/preflight/verify_migrations.py
```

All three must report GO / 9-migration chain / 0 pending. If any
fail: HALT, copy logs, do not auto-retry.

### Step 3b — Execute deploy

```bash
cd services/api
flyctl deploy --app vlabs-api --strategy rolling --wait-timeout 600 \
    | tee /tmp/flyctl_deploy.log
```

Strategy `rolling` means Fly replaces machines one at a time so the
service stays up during the deploy. `--wait-timeout 600` (10 min) is
generous — typical deploys land in 2-3 minutes.

The container's `entrypoint.sh` runs `alembic upgrade head` before
launching uvicorn, so migrations 0001-0009 are applied automatically
on first machine boot.

### Step 3c — Post-deploy verification

```bash
# Health endpoint (returns migration version + service status)
curl -sf https://api.verifiable-labs.com/v1/health

# OpenAPI surface (smoke check — should expose all 33 endpoints)
curl -sf https://api.verifiable-labs.com/openapi.json | jq '.paths | keys | length'
# Expect: 33 (Phase 22 + 23 + 28 + 29 + 30 + 31 + management)
```

CC also runs an automated post-deploy smoke test:

```bash
python scripts/deploy/post_deploy_smoke.py \
    --base-url https://api.verifiable-labs.com
```

(Generated in Phase 3 — exercises one endpoint per phase plane.)

### Step 3d — Migration final verification via Supabase MCP

CC uses the Supabase MCP (when wired) to confirm production schema:

```
Supabase: list_migrations
# Expect: 0001 → 0009 chain

Supabase: list_tables
# Expect: api_keys, users, calibration_runs, evaluations,
# audit_calls, dataset_jobs, monitors, monitor_runs,
# monitor_alerts, reward_models, reward_model_runs,
# process_reward_models, process_reward_model_runs,
# attestations, attestation_artifacts, attestation_audits,
# attestation_renewals, attestation_certificates,
# usage_counters, subscriptions, stripe_events
```

### Step 3e — Final commit

CC commits `reports/deploy/POST_DEPLOY_REPORT.md` with the
consolidated deploy outcome (timing, endpoint coverage, migration
status, health green/red).

Subject: `deploy: production rollout phase 22-31 to fly.io`.

## Failure recovery

| Failure mode                          | Response                                  |
|---------------------------------------|-------------------------------------------|
| `verify_deploy_readiness.sh` NO-GO    | Re-run provisioner; check missing keys.   |
| Alembic migration error on boot       | Check `flyctl logs --app vlabs-api`; do not auto-rollback. |
| `/v1/health` returns non-200          | HALT; collect logs; rollback via `flyctl releases list` + `flyctl deploy --image <previous>`. |
| Per-endpoint smoke fails              | HALT; investigate the specific endpoint family before any further deploys. |
| Fly.io secret out of sync             | `flyctl secrets set NAME=value --app vlabs-api && flyctl deploy --strategy immediate`. |

CC never auto-rollbacks — rollback is a Stelios decision based on
impact (live customer traffic vs internal smoke).

## Security guarantees

- No secret value ever appears in chat. Provisioner uses `read -srp`
  (silent prompts).
- `services/api/.env.local` is in `.gitignore` and chmod 600.
- The provisioner refuses to write if `.env.local` is NOT
  gitignored.
- Each prompt zeros the value variable + best-effort removes the
  prompt from shell history (`history -d`).
- Fly.io secrets are encrypted at rest by Fly; values are never
  printed by `flyctl secrets list` (only key names + last-updated).
- Generated `POST_DEPLOY_REPORT.md` does NOT contain any secret
  value (only redacted DSN, key names, status flags).

## Out of scope

- Stripe live mode (gated on Delaware C-corp; tracked in
  `PHASE_31_PLAN.md` §19).
- Phase 31.G first-customer-acquisition launch.
- GPU training (Phase 29.F + 30.F — separate session prompt).
