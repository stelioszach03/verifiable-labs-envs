#!/bin/bash
# First-time deploy script for vlabs-api on Fly.io.
#
# Reads every secret from your local services/api/.env.local — never
# from CLI args, never echoed. Pipes them into `fly secrets set`.
#
# Run once after:
#   - flyctl auth login
#   - DATABASE_URL set in .env.local (Supabase pooler URL)
#   - Other vars set per the audit
#
# Idempotent up to a point: re-running re-uploads secrets and triggers
# a new deploy, but `fly launch` is skipped if the app already exists.
#
#   ./services/api/deploy/first-deploy.sh

set -euo pipefail

APP_NAME="vlabs-api"
REGION="iad"
HERE="$(cd "$(dirname "$0")" && pwd)"
API_ROOT="$(cd "$HERE/.." && pwd)"
ENV_LOCAL="$API_ROOT/.env.local"

if [ ! -f "$ENV_LOCAL" ]; then
    echo "[deploy] ERROR: $ENV_LOCAL not found" >&2
    exit 1
fi

# Make sure flyctl + auth are present.
if ! command -v flyctl >/dev/null; then
    echo "[deploy] ERROR: flyctl not in PATH" >&2
    exit 1
fi
flyctl auth whoami >/dev/null || { echo "[deploy] ERROR: not logged in" >&2; exit 1; }

cd "$API_ROOT"

# Step 1 — create the app (no-deploy so we can set secrets first).
if ! flyctl status --app "$APP_NAME" >/dev/null 2>&1; then
    echo "[deploy] creating app $APP_NAME in $REGION"
    flyctl launch --no-deploy --copy-config --name "$APP_NAME" --region "$REGION" --org personal
else
    echo "[deploy] app $APP_NAME already exists, skipping launch"
fi

# Step 2 — push secrets from .env.local (key=value lines only, comments stripped).
echo "[deploy] uploading secrets (values never echoed)"

SECRET_KEYS=(
    DATABASE_URL
    VLABS_API_KEY_HASH_PEPPER
    VLABS_BILLING_ENABLED
    VLABS_ADMIN_CLERK_IDS
    CLERK_SECRET_KEY
    CLERK_PUBLISHABLE_KEY
    CLERK_JWT_ISSUER
    CLERK_JWKS_URL
    SENTRY_DSN
    SENTRY_TRACES_SAMPLE_RATE
    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN
    BETTERSTACK_API_TOKEN
    STRIPE_SECRET_KEY
    STRIPE_WEBHOOK_SECRET
    STRIPE_PRICE_ID_PRO
    STRIPE_PRICE_ID_TEAM
    STRIPE_PRICE_ID_PRO_OVERAGE
    STRIPE_PRICE_ID_TEAM_OVERAGE
)

ARGS=()
for k in "${SECRET_KEYS[@]}"; do
    # Extract value without sourcing — works with passwords containing *, (, $, etc.
    v=$(awk -F= -v k="$k" 'BEGIN{IGNORECASE=0} $1==k {sub("^"k"=",""); print; exit}' "$ENV_LOCAL")
    # Strip optional surrounding quotes
    v="${v%\"}"; v="${v#\"}"
    v="${v%\'}"; v="${v#\'}"
    if [ -n "$v" ]; then
        ARGS+=("$k=$v")
    fi
done

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "[deploy] ERROR: no secrets parsed from .env.local" >&2
    exit 1
fi

flyctl secrets set --app "$APP_NAME" --stage "${ARGS[@]}"

# Step 3 — deploy.
echo "[deploy] deploying image"
flyctl deploy --app "$APP_NAME" --remote-only

# Step 4 — TLS cert for the custom hostname.
echo "[deploy] requesting TLS cert for api.verifiable-labs.com"
flyctl certs create --app "$APP_NAME" "api.verifiable-labs.com" || \
    echo "[deploy]   (cert already exists or DNS not yet pointed; safe to retry)"

# Step 5 — sanity check.
echo "[deploy] hitting /health"
sleep 5
flyctl status --app "$APP_NAME"
curl -fsSL "https://${APP_NAME}.fly.dev/health" || \
    echo "[deploy]   /health did not respond — check `flyctl logs --app $APP_NAME`"

echo "[deploy] done."
echo "         Custom domain: https://api.verifiable-labs.com (after DNS propagates)"
echo "         Direct:        https://${APP_NAME}.fly.dev"
