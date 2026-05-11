#!/usr/bin/env bash
# scripts/preflight/_save_provider_secrets.sh — persist cloud-provider
# API tokens (Vultr + Oracle, plus optional DigitalOcean / RunPod /
# Thunder / Modal) to a chmod-600 file under ~/.vlabs-secrets/ so the
# training-launch automation can read them without re-prompting.
#
# Counterpart of:
#   scripts/preflight/_load_provider_secrets.sh — session-only loader
#                                                  (doesn't write to
#                                                  disk; for shells
#                                                  that already have
#                                                  the tokens exported)
#
# Mirrors _save_remote_publish_tokens.sh + _save_training_secrets.sh:
#   - each prompt is silent (``read -srp``) — no echo, no shell history;
#   - writes to $HOME/.vlabs-secrets/provider-tokens.env (outside the
#     repo tree);
#   - chmod 600 on the file + chmod 700 on the directory;
#   - idempotent — Enter at a prompt keeps the previously-saved value.
#
# What you need before running:
#
#   VULTR_API_KEY              — https://my.vultr.com/settings/#settingsapi
#                                 (Personal Access Token; needs Bare-Metal
#                                 + Compute scope for A100 provisioning)
#
#   ORACLE_CLI_AUTH_TOKEN      — Easiest: an OCI CLI auth token from
#                                 https://cloud.oracle.com/identity/users/<user>/auth-tokens
#                                 (works as "Bearer" for the OCI REST API)
#                                 OR pair of OCID + private key (the four
#                                 OCID-prefixed fields below) for full
#                                 keypair-signed requests.
#
#   ORACLE_TENANCY_OCID        — ocid1.tenancy.oc1..XXXX (find at
#                                 https://cloud.oracle.com → Profile → Tenancy)
#
#   ORACLE_USER_OCID           — ocid1.user.oc1..XXXX (Profile → User Settings)
#
#   ORACLE_FINGERPRINT         — e.g. "12:34:56:78:9a:bc:de:f0:..." (from
#                                 the API-key upload step)
#
#   ORACLE_PRIVATE_KEY_PATH    — local path to the matching PEM file
#                                 (typically ~/.oci/oci_api_key.pem)
#
# Optional (skip with empty Enter if not signing up for those):
#
#   DIGITALOCEAN_API_TOKEN     — https://cloud.digitalocean.com/account/api/tokens
#   RUNPOD_API_KEY             — https://www.runpod.io/console/user/settings
#   THUNDER_API_KEY            — Thunder Compute dashboard
#   MODAL_TOKEN_ID + MODAL_TOKEN_SECRET — Modal CLI auth pair
#
# Usage:
#   bash scripts/preflight/_save_provider_secrets.sh
#
# To rotate / forget:
#   rm -f ~/.vlabs-secrets/provider-tokens.env

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: must be run via bash (not sh)." >&2
    exit 2
fi

SECRETS_DIR="$HOME/.vlabs-secrets"
SECRETS_FILE="$SECRETS_DIR/provider-tokens.env"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# ── load existing values (if any) ─────────────────────────────────
declare -A EXISTING=()
if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$SECRETS_FILE"; set +a
fi
for key in VULTR_API_KEY ORACLE_CLI_AUTH_TOKEN ORACLE_TENANCY_OCID \
           ORACLE_USER_OCID ORACLE_FINGERPRINT ORACLE_PRIVATE_KEY_PATH \
           DIGITALOCEAN_API_TOKEN RUNPOD_API_KEY THUNDER_API_KEY \
           MODAL_TOKEN_ID MODAL_TOKEN_SECRET; do
    EXISTING[$key]="${!key:-}"
done

echo "Verifiable Labs — provider token persister"
echo ""
echo "Each prompt is silent (cursor but no keystroke echo)."
echo "Press Enter without typing to KEEP the previously-saved value."
echo "Press Enter twice to leave a field empty (optional providers)."
echo ""

# Helper: read silently, fall back to existing on empty input
prompt_for() {
    local key="$1" label="$2" existing="${EXISTING[$key]}"
    local prompt
    if [ -n "$existing" ]; then
        prompt="  ${label} [existing length=${#existing}]"
    else
        prompt="  ${label}"
    fi
    local new_value
    read -srp "$prompt > " new_value || new_value=""
    echo
    if [ -z "$new_value" ]; then
        printf '%s' "$existing"
    else
        printf '%s' "$new_value"
    fi
}

# ── primary required fields ───────────────────────────────────────
VULTR=$(prompt_for VULTR_API_KEY "Vultr API key (required for E1-E5)")
echo "  ── Oracle Cloud (required for E6-E10):"
ORACLE_TOKEN=$(prompt_for ORACLE_CLI_AUTH_TOKEN "Oracle CLI auth token (optional if using OCID+keypair)")
ORACLE_TEN=$(prompt_for ORACLE_TENANCY_OCID "Oracle tenancy OCID")
ORACLE_USR=$(prompt_for ORACLE_USER_OCID "Oracle user OCID")
ORACLE_FP=$(prompt_for ORACLE_FINGERPRINT "Oracle API-key fingerprint")
ORACLE_PEM=$(prompt_for ORACLE_PRIVATE_KEY_PATH "Oracle private-key path (e.g. ~/.oci/oci_api_key.pem)")

# ── optional secondary providers ──────────────────────────────────
echo ""
echo "  ── optional providers (Enter to skip):"
DO_TOKEN=$(prompt_for DIGITALOCEAN_API_TOKEN "DigitalOcean API token")
RUNPOD=$(prompt_for RUNPOD_API_KEY "RunPod API key")
THUNDER=$(prompt_for THUNDER_API_KEY "Thunder Compute API key")
MODAL_ID=$(prompt_for MODAL_TOKEN_ID "Modal token ID")
MODAL_SEC=$(prompt_for MODAL_TOKEN_SECRET "Modal token secret")

# ── validate at least one provider has credentials ────────────────
if [ -z "$VULTR" ] && [ -z "$ORACLE_TOKEN" ] && [ -z "$ORACLE_TEN" ]; then
    echo "" >&2
    echo "ERROR: at least one of VULTR_API_KEY or ORACLE_* must be set." >&2
    exit 2
fi

# ── write atomically with chmod 600 ───────────────────────────────
TMP="$(mktemp "${SECRETS_DIR}/.provider-tokens.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
{
    echo "# Provider credentials — written by"
    echo "# scripts/preflight/_save_provider_secrets.sh"
    echo "# DO NOT commit. chmod 600 enforced below."
    echo ""
    echo "# === Primary GPU providers ==="
    echo "VULTR_API_KEY=$VULTR"
    echo ""
    echo "ORACLE_CLI_AUTH_TOKEN=$ORACLE_TOKEN"
    echo "ORACLE_TENANCY_OCID=$ORACLE_TEN"
    echo "ORACLE_USER_OCID=$ORACLE_USR"
    echo "ORACLE_FINGERPRINT=$ORACLE_FP"
    echo "ORACLE_PRIVATE_KEY_PATH=$ORACLE_PEM"
    echo ""
    echo "# === Optional secondary providers ==="
    echo "DIGITALOCEAN_API_TOKEN=$DO_TOKEN"
    echo "RUNPOD_API_KEY=$RUNPOD"
    echo "THUNDER_API_KEY=$THUNDER"
    echo "MODAL_TOKEN_ID=$MODAL_ID"
    echo "MODAL_TOKEN_SECRET=$MODAL_SEC"
} > "$TMP"
chmod 600 "$TMP"
mv "$TMP" "$SECRETS_FILE"
trap - EXIT

# ── confirm without echoing values ────────────────────────────────
echo ""
echo "  ✓ Saved to $SECRETS_FILE (chmod 600)"
echo ""
printf "    VULTR_API_KEY               length=%d\n" "${#VULTR}"
printf "    ORACLE_CLI_AUTH_TOKEN       length=%d\n" "${#ORACLE_TOKEN}"
printf "    ORACLE_TENANCY_OCID         length=%d\n" "${#ORACLE_TEN}"
printf "    ORACLE_USER_OCID            length=%d\n" "${#ORACLE_USR}"
printf "    ORACLE_FINGERPRINT          length=%d\n" "${#ORACLE_FP}"
printf "    ORACLE_PRIVATE_KEY_PATH     length=%d\n" "${#ORACLE_PEM}"
printf "    DIGITALOCEAN_API_TOKEN      length=%d\n" "${#DO_TOKEN}"
printf "    RUNPOD_API_KEY              length=%d\n" "${#RUNPOD}"
printf "    THUNDER_API_KEY             length=%d\n" "${#THUNDER}"
printf "    MODAL_TOKEN_ID              length=%d\n" "${#MODAL_ID}"
printf "    MODAL_TOKEN_SECRET          length=%d\n" "${#MODAL_SEC}"
echo ""
echo "Next steps for the training-launch flow:"
echo "  1. Generate cloud SSH keypair (if not already):"
echo "       ssh-keygen -t ed25519 -f ~/.ssh/vlabs_cloud_id -N ''"
echo "  2. Verify provider liveness:"
echo "       bash -c 'set -a; . ~/.vlabs-secrets/provider-tokens.env; set +a; \\"
echo "                python3 scripts/preflight/provider_status.py --only vultr'"
