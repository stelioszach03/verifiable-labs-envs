#!/usr/bin/env bash
# scripts/training/check_training_secrets.sh — read-only inventory of
# the secrets needed by the Tier 1-4 training + dataset-prep pipeline.
#
# NEVER prints values; only key names + lengths.
#
# Sources, in order:
#   - services/api/.env.local (production deploy secrets — already
#     provisioned via scripts/deploy/provision_production_secrets.sh)
#   - ~/.vlabs-secrets/training-secrets.env (saved via _save_training_secrets.sh)
#   - ~/.vlabs-secrets/pypi-tokens.env (saved via _save_pypi_tokens.sh)

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

[ -f services/api/.env.local ] && set -a && . services/api/.env.local && set +a
[ -f "$HOME/.vlabs-secrets/training-secrets.env" ] && set -a && . "$HOME/.vlabs-secrets/training-secrets.env" && set +a
[ -f "$HOME/.vlabs-secrets/pypi-tokens.env" ] && set -a && . "$HOME/.vlabs-secrets/pypi-tokens.env" && set +a

report_one() {
    local key="$1"
    local note="$2"
    local val="${!key:-}"
    if [ -n "$val" ]; then
        printf "  ✓ %-32s (length=%d)   %s\n" "$key" "${#val}" "$note"
    else
        printf "  ✗ %-32s MISSING            %s\n" "$key" "$note"
    fi
}

echo "=== Tier 1-4 training secrets ==="
report_one "OPENROUTER_API_KEY"   "(T2.6 + T2.7 + T4.1 + T4.2 frontier judgments)"
report_one "HF_TOKEN"             "(RewardBench + ProcessBench + dataset uploads)"
report_one "WANDB_API_KEY"        "(optional — run tracking)"
echo
echo "=== Production deploy secrets (already provisioned) ==="
report_one "DATABASE_URL"                 "(Supabase Postgres)"
report_one "CLERK_SECRET_KEY"             "(Clerk dashboard auth)"
report_one "CLERK_PUBLISHABLE_KEY"        "(Clerk frontend public)"
report_one "VLABS_API_KEY_HASH_PEPPER"    "(API key hashing)"
report_one "VLABS_R2_ACCOUNT_ID"          "(Cloudflare R2)"
report_one "VLABS_R2_ACCESS_KEY_ID"       "(Cloudflare R2)"
report_one "VLABS_R2_SECRET_ACCESS_KEY"   "(Cloudflare R2)"
report_one "VLABS_DATA_LLM_KEY_ENCRYPTION" "(Fernet key for dataset_jobs)"
report_one "VLABS_EMAIL_API_KEY"          "(Resend transactional email)"
echo
echo "=== PyPI publish secrets ==="
report_one "PYPI_API_TOKEN"          "(prod pypi.org)"
report_one "TEST_PYPI_API_TOKEN"     "(test.pypi.org)"
echo
echo "Sources scanned (in order, later overrides earlier):"
echo "  1. services/api/.env.local"
echo "  2. ~/.vlabs-secrets/training-secrets.env"
echo "  3. ~/.vlabs-secrets/pypi-tokens.env"
