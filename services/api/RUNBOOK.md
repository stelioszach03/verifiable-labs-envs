# vlabs-api — Incident Response Runbook

Production incidents: what to check, in what order, with the exact commands.

> **Page yourself first**: BetterStack monitor on `/health` should already
> have alerted you. If you got here from Sentry, scroll the breadcrumbs first
> — most issues fingerprint cleanly.

## 1. Database is down (asyncpg connection errors)

**Symptoms**: Sentry floods with `OperationalError`, `gaierror`, or
`InternalServerError: Tenant or user not found`.
`/health` may still return 200 (it doesn't hit the DB by design).
`/v1/usage` returns 500.

**Diagnose**:

```bash
flyctl logs --app vlabs-api | grep -i 'database\|asyncpg\|operational'
# In a separate terminal, hit Supabase:
curl -s https://uvrljxgjocxcbehzqseu.supabase.co/rest/v1/ \
     -H "apikey: $(grep ^NEXT_PUBLIC_SUPABASE_ANON_KEY services/landing/.env.local | cut -d= -f2-)"
# Expect: 200 OK with project metadata.
```

If Supabase REST is down too → wait for Supabase status page (status.supabase.com).

**Common fix**: pooler endpoint changed (we hit `aws-1-us-west-1.pooler...` —
verify in Supabase dashboard → Settings → Database → "Transaction Pooler").

```bash
# Update DATABASE_URL secret (don't echo the value):
flyctl secrets set --app vlabs-api DATABASE_URL="postgresql+asyncpg://postgres.<ref>:<pw>@<new-host>:6543/postgres"
# Fly auto-restarts machines.
```

## 2. Rate-limit spikes (429 floods)

**Symptoms**: Sentry shows many `RateLimited` exceptions; user reports
"I'm getting 429 even though I just upgraded to Pro".

**Diagnose**:

```bash
flyctl logs --app vlabs-api | grep -i 'rate_limited\|tier='
# Check if the user's tier resolved correctly:
flyctl ssh console --app vlabs-api
> python -c "
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
import os, asyncio
from vlabs_api.db import Subscription, User
e = create_async_engine(os.environ['DATABASE_URL'])
SF = async_sessionmaker(e, expire_on_commit=False)
async def go():
    async with SF() as s:
        rows = (await s.execute(select(User, Subscription).join(Subscription, User.id==Subscription.user_id, isouter=True).where(User.email=='customer@example.com'))).all()
        print(rows)
asyncio.run(go())
"
```

**Common causes + fixes**:

| Cause | Fix |
|---|---|
| Stripe webhook never fired → no `subscriptions` row → tier resolves `free` | Replay the missed event from Stripe Dashboard → Developers → Events → Resend, or insert the subscription row manually with the right tier |
| Redis backend partial failure (`open the gate` log line) | Check Upstash dashboard, rotate `UPSTASH_REDIS_REST_TOKEN` if needed |
| Genuine abuse | Review the API-key prefix in logs; revoke via `/v1/keys/{id}` DELETE or directly in DB |

## 3. Memory pressure / OOM kills

**Symptoms**: Fly auto-restarts machines, `flyctl status` shows recent restarts.

**Diagnose**:

```bash
flyctl status --app vlabs-api
flyctl logs --app vlabs-api | grep -i 'oom\|killed\|memory'
```

**Fix**:

```bash
# Bump memory on the machine class:
flyctl scale memory 2048 --app vlabs-api
# Or scale out:
flyctl scale count 2 --app vlabs-api
```

Calibration with very large trace batches (>500K) can spike memory. The API
caps `traces` at 1M; if a customer hits that legitimately, consider streaming
or chunking — that's a Phase 17 product change, not an ops fix.

## 4. SSL cert renewal failed

**Symptoms**: BetterStack alert "TLS cert expires in 7 days".

**Fix**:

```bash
flyctl certs check --app vlabs-api api.verifiable-labs.com
# If renewal stalled:
flyctl certs remove --app vlabs-api api.verifiable-labs.com
flyctl certs create --app vlabs-api api.verifiable-labs.com
# Verify the DNS records still match what fly expects.
```

## 5. Webhook handler erroring (only relevant once billing enabled)

**Symptoms**: `stripe_events.error` rows appearing.

**Diagnose**:

```sql
SELECT event_id, event_type, error, received_at
FROM stripe_events
WHERE error IS NOT NULL
ORDER BY received_at DESC
LIMIT 20;
```

**Fix**:

- If the error is `KeyError` on the event payload → Stripe schema changed.
  Update `vlabs_api/billing.py:sync_subscription_from_event`.
- If the error is `IntegrityError` on `subscriptions.user_id_fkey` → the
  `users.stripe_customer_id` link wasn't set. Find the customer in Stripe,
  back-fill `users.stripe_customer_id`, then "Resend" the event from the Stripe
  Dashboard.

## 6. Clerk JWT verification failing

**Symptoms**: 401 `invalid_clerk_token` for every dashboard request.

**Diagnose**:

```bash
flyctl logs --app vlabs-api | grep -i 'clerk\|jwt\|jwks'
curl -fsSL "$CLERK_JWT_ISSUER/.well-known/jwks.json" | jq '.keys | length'
```

**Common cause**: Clerk rotated their signing keys; our cached PyJWKClient is
stale. The cache TTL is 1h, so a restart clears it:

```bash
flyctl machine restart --app vlabs-api
```

## 7. Catastrophic — full outage

If the API is fully unreachable and you need to fail fast:

1. **Status page**: post an incident on `status.verifiable-labs.com`
   (BetterStack lets you draft from a phone).
2. **Roll back**: `flyctl releases --app vlabs-api` → pick last green →
   `flyctl deploy --image registry.fly.io/vlabs-api:v<N>`.
3. **DM**: send a Twitter post + email customers. Template lives in
   `services/api/docs/incident-template.md` (TBD — Phase 17).

## On-call hand-off (future)

When we have an actual on-call rotation, this section becomes a one-page
hand-off doc. For now: Stelios is on-call, BetterStack pings the email
in his Clerk profile.
