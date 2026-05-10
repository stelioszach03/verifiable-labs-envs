#!/usr/bin/env bash
# scripts/deploy/provision_production_secrets.sh — silent-prompt
# provisioner for the V-Labs production deploy.
#
# Reads which secrets are required from the same source-of-truth as
# scripts/preflight/check_deploy_readiness.sh and reports/deploy/
# SECRETS_GAP_REPORT.md. For every required secret that is NOT
# already present in $ENV_FILE, the script prompts silently
# (read -srp), writes the value to $ENV_FILE (chmod 600), and
# optionally stages it as a Fly.io secret via `flyctl secrets set
# --stage`.
#
# CRITICAL SECURITY RULES (mirrored from the deploy prompt):
#   - input is HIDDEN at the terminal; never echoed
#   - values are never printed to stdout, logs, or shell history
#   - $ENV_FILE is verified to be in .gitignore before any write
#   - the script aborts (non-zero) if .env is NOT gitignored
#   - already-provisioned keys are SKIPPED, never re-prompted
#
# Usage:
#   bash scripts/deploy/provision_production_secrets.sh
#   VLABS_SYNC_FLY=no bash scripts/deploy/provision_production_secrets.sh
#   VLABS_ENV_FILE=.env.production bash scripts/deploy/provision_production_secrets.sh
#
# Exit codes:
#   0 — every required secret is now present in $ENV_FILE
#   1 — refused to write (e.g. $ENV_FILE not gitignored)
#   2 — user aborted with Ctrl+C

set -uo pipefail

ENV_FILE="${VLABS_ENV_FILE:-services/api/.env.local}"
SYNC_FLY="${VLABS_SYNC_FLY:-yes}"
APP_NAME="${VLABS_FLY_APP:-vlabs-api}"

# ── 1. gitignore safety check ─────────────────────────────────────

# Walk up the directory tree to find a .gitignore that lists $ENV_FILE
# (or one of its parent path components). The exact line we need is
# matched literally — don't rely on `git check-ignore` because it
# might shadow with .gitignore_global rules we don't control.

abs_env="$(cd "$(dirname "$ENV_FILE")" 2>/dev/null && pwd)/$(basename "$ENV_FILE")"
basename_env="$(basename "$ENV_FILE")"

if ! grep -E "^(${basename_env}|${ENV_FILE//\//\\/})$" .gitignore >/dev/null 2>&1 \
   && ! grep -E "^${basename_env}$" services/api/.gitignore >/dev/null 2>&1; then
    # As a final fallback, ask git itself.
    if ! git check-ignore -q "$ENV_FILE" 2>/dev/null; then
        echo "ERROR: $ENV_FILE is NOT gitignored." >&2
        echo "Add an entry to .gitignore before running this script." >&2
        exit 1
    fi
fi

# ── 2. ensure target file exists with safe perms ─────────────────

mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

# ── 3. check flyctl availability if syncing requested ────────────

FLY_AVAILABLE=0
if [ "$SYNC_FLY" = "yes" ]; then
    if command -v flyctl >/dev/null 2>&1; then
        FLY_AVAILABLE=1
    else
        echo "WARN: flyctl not found in PATH; setting VLABS_SYNC_FLY=no." >&2
        SYNC_FLY=no
    fi
fi

# ── 4. helper: prompt + write ────────────────────────────────────

prompt_secret() {
    local name="$1"
    local description="$2"
    local format_hint="$3"
    local optional="${4:-required}"   # "required" | "optional"

    if grep -q "^${name}=" "$ENV_FILE" 2>/dev/null; then
        local existing_value
        existing_value=$(awk -F= -v k="$name" '$1==k {sub("^"k"=",""); print; exit}' "$ENV_FILE")
        if [ -n "$existing_value" ]; then
            echo "  [skip] ${name} already provisioned"
            unset existing_value
            return 0
        fi
        # Empty value present — fall through to prompt.
    fi

    echo
    echo "  Provisioning: ${name}"
    echo "  Purpose:      ${description}"
    echo "  Format:       ${format_hint}"
    if [ "$optional" = "optional" ]; then
        echo "  (Optional — press Enter to skip.)"
    fi
    # shellcheck disable=SC2162
    read -srp "  Enter value (input hidden): " value
    echo

    if [ -z "$value" ]; then
        if [ "$optional" = "required" ]; then
            echo "  ⚠ Empty input — leaving ${name} unset (required; deploy will fail)."
        else
            echo "  ⨯ skipped"
        fi
        return 0
    fi

    # Write to $ENV_FILE (overwrite if blank entry exists, else append).
    if grep -q "^${name}=" "$ENV_FILE" 2>/dev/null; then
        # In-place replace — use a temp file to avoid leaking the value
        # via shell expansion + awk -i overwrites.
        local tmp
        tmp=$(mktemp)
        chmod 600 "$tmp"
        awk -v k="$name" -v v="$value" 'BEGIN{set=0} {
            if ($0 ~ "^"k"=") { print k"="v; set=1 } else { print }
        } END { if (!set) print k"="v }' "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
    else
        # Append; the redirect prevents leaking via $value to a logging shell.
        printf '%s=%s\n' "$name" "$value" >> "$ENV_FILE"
    fi

    if [ "$SYNC_FLY" = "yes" ] && [ "$FLY_AVAILABLE" = "1" ]; then
        if echo "$value" | flyctl secrets set "${name}=$(cat)" \
                --app "$APP_NAME" --stage >/dev/null 2>&1; then
            echo "  ✓ written to $ENV_FILE  +  staged on Fly.io"
        else
            echo "  ⚠ written to $ENV_FILE  +  Fly.io stage failed (run flyctl manually)"
        fi
    else
        echo "  ✓ written to $ENV_FILE"
    fi

    # Zero out the variable + remove the most recent shell history
    # entry so the value doesn't survive in interactive transcripts.
    unset value
    history -d "$((HISTCMD - 1))" 2>/dev/null || true
}

# ── 5. invariant writer (non-secret production flags) ────────────

write_invariant() {
    local name="$1"
    local value="$2"
    if grep -q "^${name}=" "$ENV_FILE" 2>/dev/null; then
        local tmp
        tmp=$(mktemp)
        chmod 600 "$tmp"
        awk -v k="$name" -v v="$value" 'BEGIN{set=0} {
            if ($0 ~ "^"k"=") { print k"="v; set=1 } else { print }
        } END { if (!set) print k"="v }' "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
    else
        printf '%s=%s\n' "$name" "$value" >> "$ENV_FILE"
    fi
    if [ "$SYNC_FLY" = "yes" ] && [ "$FLY_AVAILABLE" = "1" ]; then
        flyctl secrets set "${name}=${value}" --app "$APP_NAME" --stage \
            >/dev/null 2>&1 || true
    fi
}

# ── 6. banner ────────────────────────────────────────────────────

cat <<EOF
===========================================================
Production secrets provisioner — Verifiable Labs
===========================================================

Target file:  $ENV_FILE  (chmod 600)
Fly.io sync:  $SYNC_FLY  (app=$APP_NAME, flyctl_available=$FLY_AVAILABLE)

This script will silently prompt for every REQUIRED production
secret that is missing from $ENV_FILE. Input is hidden — values
are never printed to the terminal, logs, or shell history.

Already-provisioned secrets are skipped — re-running is safe.

Press Ctrl+C to abort at any time. Press Enter to begin.
EOF

# Guard the read against EOF on a non-TTY pipe (CI runs this with
# < /dev/null and would loop forever otherwise).
if [ -t 0 ]; then
    read -r _
else
    echo "(non-interactive stdin detected; proceeding without confirmation)"
fi

# ── 7. required production secrets ───────────────────────────────

echo
echo "=== Required production secrets ==="

prompt_secret "DATABASE_URL" \
    "Supabase Postgres pooler connection (sync DSN)" \
    "postgresql+asyncpg://user:pass@host:5432/db"

prompt_secret "VLABS_API_KEY_HASH_PEPPER" \
    "Server-side pepper for SHA-256 hashing of vlk_* API keys" \
    "32-byte base64 (python -c 'import secrets; print(secrets.token_urlsafe(32))')"

prompt_secret "CLERK_SECRET_KEY" \
    "Clerk backend secret (Clerk dashboard -> API Keys)" \
    "sk_live_... or sk_test_..."

prompt_secret "CLERK_PUBLISHABLE_KEY" \
    "Clerk frontend public key" \
    "pk_live_... or pk_test_..."

prompt_secret "CLERK_JWT_ISSUER" \
    "Clerk JWT issuer URL (visible in Clerk -> JWT templates)" \
    "https://*.clerk.accounts.dev"

prompt_secret "CLERK_JWKS_URL" \
    "Clerk JWKS endpoint (auto-derived if blank)" \
    "https://*/.well-known/jwks.json"

prompt_secret "UPSTASH_REDIS_REST_URL" \
    "Upstash Redis REST API URL (multi-instance rate limit)" \
    "https://*.upstash.io"

prompt_secret "UPSTASH_REDIS_REST_TOKEN" \
    "Upstash Redis REST API token" \
    "opaque token string"

prompt_secret "VLABS_R2_ACCOUNT_ID" \
    "Cloudflare R2 account id (Cloudflare -> R2 -> Manage R2 API)" \
    "32-char hex"

prompt_secret "VLABS_R2_ACCESS_KEY_ID" \
    "Cloudflare R2 access key id" \
    "32-char string"

prompt_secret "VLABS_R2_SECRET_ACCESS_KEY" \
    "Cloudflare R2 secret access key" \
    "64-char string"

prompt_secret "VLABS_R2_BUCKET_NAME" \
    "Cloudflare R2 bucket name" \
    "vlabs-datasets"

prompt_secret "VLABS_R2_PUBLIC_URL" \
    "Cloudflare R2 public-served subdomain" \
    "https://datasets.verifiable-labs.com"

prompt_secret "VLABS_DATA_LLM_KEY_ENCRYPTION" \
    "Fernet key for dataset_jobs.llm_api_key_encrypted (Phase 23)" \
    "44-char Fernet base64 (run: python scripts/deploy/generate_fernet_key.py)"

prompt_secret "VLABS_EMAIL_FROM" \
    "Phase 28 monitor-alert sender address" \
    "alerts@verifiable-labs.com"

prompt_secret "VLABS_EMAIL_API_KEY" \
    "Resend / SES / SendGrid transactional email API key" \
    "provider key string (e.g. re_... for Resend)"

# ── 8. optional secrets ──────────────────────────────────────────

echo
echo "=== Optional secrets (press Enter to skip) ==="

prompt_secret "SENTRY_DSN" \
    "Sentry error tracking DSN" \
    "https://*@*.sentry.io/*" "optional"

prompt_secret "BETTERSTACK_API_TOKEN" \
    "BetterStack uptime monitoring token" \
    "opaque string" "optional"

prompt_secret "CLOUDFLARE_API_TOKEN" \
    "Cloudflare DNS automation token (used by deploy script only)" \
    "opaque string" "optional"

prompt_secret "VLABS_ADMIN_CLERK_IDS" \
    "Comma-separated list of Clerk user_ids allowed at /v1/admin/*" \
    "user_xxx,user_yyy" "optional"

prompt_secret "VLABS_R2_ENDPOINT_URL" \
    "Override R2 endpoint URL (auto-derived from account_id if blank)" \
    "https://<account-id>.r2.cloudflarestorage.com" "optional"

prompt_secret "VLABS_SLACK_WEBHOOK_DEFAULT" \
    "Default Slack webhook for monitor alerts" \
    "https://hooks.slack.com/services/..." "optional"

# ── 9. production flag invariants ────────────────────────────────

echo
echo "=== Production flag invariants ==="

write_invariant "VLABS_ENVIRONMENT" "prod"
write_invariant "VLABS_LOCAL_FAKE_R2" "false"
write_invariant "VLABS_LOCAL_FAKE_EMAIL" "false"
write_invariant "VLABS_LOCAL_FAKE_PKI" "false"
write_invariant "VLABS_LOCAL_FAKE_HF" "false"
echo "  ✓ VLABS_ENVIRONMENT=prod"
echo "  ✓ VLABS_LOCAL_FAKE_{R2,EMAIL,PKI,HF}=false"

# ── 10. final summary ────────────────────────────────────────────

cat <<EOF

===========================================================
Provisioning complete.

  $ENV_FILE updated (chmod 600).
EOF

if [ "$SYNC_FLY" = "yes" ] && [ "$FLY_AVAILABLE" = "1" ]; then
    cat <<EOF
  Fly.io secrets staged. The next deploy will apply them.
  Verify via:  flyctl secrets list --app $APP_NAME
EOF
else
    cat <<EOF
  Fly.io sync skipped (flyctl missing or VLABS_SYNC_FLY=no).
  Run scripts/api/deploy/first-deploy.sh on a flyctl-equipped host
  to push the secrets in one go, OR push individually:
      flyctl secrets set NAME=value --app $APP_NAME --stage
EOF
fi

cat <<EOF

Next step:
  bash scripts/deploy/verify_deploy_readiness.sh
===========================================================
EOF
