#!/usr/bin/env bash
# scripts/deploy/verify_deploy_readiness.sh — go/no-go gate before
# `flyctl deploy --app vlabs-api`.
#
# Verifies that EVERY required production secret is present in BOTH
# the local $ENV_FILE AND (if flyctl is available) Fly.io secrets.
# Outputs a per-key table + an overall VERDICT (GO / NO-GO).
#
# This script never reads or prints secret VALUES — only key names.
#
# Usage:
#   bash scripts/deploy/verify_deploy_readiness.sh
#   bash scripts/deploy/verify_deploy_readiness.sh --no-fly
#   VLABS_ENV_FILE=services/api/.env.local \
#       bash scripts/deploy/verify_deploy_readiness.sh
#
# Exit codes:
#   0 — VERDICT: GO (every required secret in both stores)
#   1 — VERDICT: NO-GO

set -uo pipefail

ENV_FILE="${VLABS_ENV_FILE:-services/api/.env.local}"
APP_NAME="${VLABS_FLY_APP:-vlabs-api}"

NO_FLY=0
for arg in "$@"; do
    case "$arg" in
        --no-fly) NO_FLY=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

REQUIRED_SECRETS=(
    DATABASE_URL
    VLABS_API_KEY_HASH_PEPPER
    CLERK_SECRET_KEY
    CLERK_PUBLISHABLE_KEY
    CLERK_JWT_ISSUER
    CLERK_JWKS_URL
    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN
    VLABS_R2_ACCOUNT_ID
    VLABS_R2_ACCESS_KEY_ID
    VLABS_R2_SECRET_ACCESS_KEY
    VLABS_R2_BUCKET_NAME
    VLABS_R2_PUBLIC_URL
    VLABS_DATA_LLM_KEY_ENCRYPTION
    VLABS_EMAIL_FROM
    VLABS_EMAIL_API_KEY
)

OPTIONAL_SECRETS=(
    SENTRY_DSN
    BETTERSTACK_API_TOKEN
    CLOUDFLARE_API_TOKEN
    VLABS_ADMIN_CLERK_IDS
    VLABS_R2_ENDPOINT_URL
    VLABS_SLACK_WEBHOOK_DEFAULT
)

INVARIANTS=(
    "VLABS_ENVIRONMENT=prod"
    "VLABS_LOCAL_FAKE_R2=false"
    "VLABS_LOCAL_FAKE_EMAIL=false"
    "VLABS_LOCAL_FAKE_PKI=false"
    "VLABS_LOCAL_FAKE_HF=false"
)

# ── helpers ──────────────────────────────────────────────────────

local_has_value() {
    local key="$1"
    [ -f "$ENV_FILE" ] || return 1
    awk -F= -v k="$key" '$1==k {sub("^"k"=",""); print; exit}' "$ENV_FILE" \
        | grep -q '[^[:space:]]'
}

local_value_equals() {
    local key="$1"
    local expected="$2"
    [ -f "$ENV_FILE" ] || return 1
    local got
    got=$(awk -F= -v k="$key" '$1==k {sub("^"k"=",""); print; exit}' "$ENV_FILE")
    [ "$got" = "$expected" ]
}

check_fly() {
    [ "$NO_FLY" = "1" ] && return 2
    command -v flyctl >/dev/null 2>&1 || return 2
    flyctl auth whoami >/dev/null 2>&1 || return 2
    return 0
}

fly_has_secret() {
    local key="$1"
    flyctl secrets list --app "$APP_NAME" 2>/dev/null \
        | awk 'NR>1 {print $1}' | grep -qx "$key"
}

# ── header ───────────────────────────────────────────────────────

echo "=== verify_deploy_readiness.sh ==="
echo "    env file:  $ENV_FILE"
echo "    fly app:   $APP_NAME"
echo

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE does not exist." >&2
    echo "Run: bash scripts/deploy/provision_production_secrets.sh" >&2
    exit 1
fi

# ── fly check ────────────────────────────────────────────────────

FLY_OK=0
case "$(check_fly; echo $?)" in
    0)
        FLY_OK=1
        echo "    flyctl:    available, authenticated"
        ;;
    *)
        echo "    flyctl:    unavailable / unauthenticated (skipping Fly checks)"
        ;;
esac
echo

# ── required secrets table ───────────────────────────────────────

REQUIRED_FAIL=0
printf "%-32s | %-7s | %-7s | %s\n" "key" "local" "fly" "status"
printf "%-32s-+-%-7s-+-%-7s-+-%s\n" "--------------------------------" "-------" "-------" "------"

for k in "${REQUIRED_SECRETS[@]}"; do
    local_ok="missing"
    fly_ok="-"
    if local_has_value "$k"; then
        local_ok="present"
    fi
    if [ "$FLY_OK" = "1" ]; then
        if fly_has_secret "$k"; then
            fly_ok="present"
        else
            fly_ok="missing"
        fi
    fi
    if [ "$local_ok" = "present" ] && { [ "$FLY_OK" = "0" ] || [ "$fly_ok" = "present" ]; }; then
        status="OK"
    else
        status="MISSING"
        REQUIRED_FAIL=$((REQUIRED_FAIL + 1))
    fi
    printf "%-32s | %-7s | %-7s | %s\n" "$k" "$local_ok" "$fly_ok" "$status"
done
echo

# ── optional secrets table (informational) ───────────────────────

echo "--- optional ---"
for k in "${OPTIONAL_SECRETS[@]}"; do
    local_ok="—"
    fly_ok="—"
    local_has_value "$k" && local_ok="present"
    if [ "$FLY_OK" = "1" ]; then
        fly_has_secret "$k" && fly_ok="present" || fly_ok="—"
    fi
    printf "%-32s | %-7s | %-7s |\n" "$k" "$local_ok" "$fly_ok"
done
echo

# ── invariants ───────────────────────────────────────────────────

echo "--- invariants ---"
INVARIANT_FAIL=0
for inv in "${INVARIANTS[@]}"; do
    key="${inv%%=*}"
    expected="${inv#*=}"
    if local_value_equals "$key" "$expected"; then
        printf "  [OK]   %s=%s\n" "$key" "$expected"
    else
        printf "  [FAIL] %s expected=%s\n" "$key" "$expected"
        INVARIANT_FAIL=$((INVARIANT_FAIL + 1))
    fi
done
echo

# ── verdict ──────────────────────────────────────────────────────

if [ "$REQUIRED_FAIL" = "0" ] && [ "$INVARIANT_FAIL" = "0" ]; then
    echo "VERDICT: GO"
    echo
    if [ "$FLY_OK" = "0" ]; then
        echo "(Fly.io coverage was not verified because flyctl is unavailable."
        echo " Re-run on a flyctl-equipped host before flyctl deploy.)"
    fi
    exit 0
else
    echo "VERDICT: NO-GO"
    echo "  required missing: $REQUIRED_FAIL"
    echo "  invariants failed: $INVARIANT_FAIL"
    echo
    echo "Run: bash scripts/deploy/provision_production_secrets.sh"
    exit 1
fi
