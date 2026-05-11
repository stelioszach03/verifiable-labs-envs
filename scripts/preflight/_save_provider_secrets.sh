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

# Helper: read silently, fall back to existing on empty input.
#
# IMPORTANT: every `echo` / `read -p` inside this function MUST be
# redirected to /dev/tty, NOT stdout. The caller is
# ``VAR=$(prompt_for ...)`` which captures stdout; if we ``echo`` the
# blank-line nicety to stdout, that newline ends up PREPENDED to the
# returned value. When the value is later written to the env file as
# ``KEY=$VAR``, the line becomes ``KEY=\nvalue`` — bash's ``source``
# then sees the value on the NEXT line and tries to execute it as a
# command (yielding ``command not found`` / ``Permission denied`` on
# the PEM path).
#
# For OCI fields, also auto-strip a leading ``prefix=`` (the OCI
# Console's "Configuration File Preview" exports lines of the form
# ``tenancy=ocid1.tenancy.oc1..XXXX`` — users frequently paste the
# whole line by reflex).
prompt_for() {
    local key="$1" label="$2" existing="${EXISTING[$key]}"
    local prompt
    if [ -n "$existing" ]; then
        prompt="  ${label} [existing length=${#existing}]"
    else
        prompt="  ${label}"
    fi
    local new_value
    # ``read -srp`` writes the prompt to stderr by default, so the
    # PROMPT text stays out of $(...) capture. But the trailing
    # newline-after-input nicety needs an explicit redirect to tty.
    read -srp "$prompt > " new_value </dev/tty 2>/dev/tty || new_value=""
    echo >/dev/tty

    local final_value
    if [ -z "$new_value" ]; then
        final_value="$existing"
    else
        final_value="$new_value"
    fi

    # Strip Windows CR (paste artifact from clipboard transitions
    # through Windows host → WSL).
    final_value="${final_value%$'\r'}"

    # OCI prefix-strip: if user pasted a full config-file line like
    # ``tenancy=ocid1...``, drop the ``<key>=`` prefix so the env var
    # contains just the OCID value.
    case "$key" in
        ORACLE_TENANCY_OCID)     final_value="${final_value#tenancy=}" ;;
        ORACLE_USER_OCID)        final_value="${final_value#user=}" ;;
        ORACLE_FINGERPRINT)      final_value="${final_value#fingerprint=}" ;;
        ORACLE_PRIVATE_KEY_PATH) final_value="${final_value#key_file=}" ;;
        ORACLE_REGION)           final_value="${final_value#region=}" ;;
    esac

    printf '%s' "$final_value"
}

# ── primary required fields ───────────────────────────────────────
VULTR=$(prompt_for VULTR_API_KEY "Vultr API key (required for E1-E5)")
echo "  ── Oracle Cloud (required for E6-E10):"
ORACLE_TOKEN=$(prompt_for ORACLE_CLI_AUTH_TOKEN "Oracle CLI auth token (optional if using OCID+keypair)")
ORACLE_TEN=$(prompt_for ORACLE_TENANCY_OCID "Oracle tenancy OCID")
ORACLE_USR=$(prompt_for ORACLE_USER_OCID "Oracle user OCID")
ORACLE_FP=$(prompt_for ORACLE_FINGERPRINT "Oracle API-key fingerprint")
ORACLE_PEM=$(prompt_for ORACLE_PRIVATE_KEY_PATH "Oracle private-key path (e.g. ~/.oci/oci_api_key.pem)")
# Region matters: OCI signature verification is region-strict and the
# Console's "Configuration File Preview" includes the user's HOME
# region. Defaulting to ``us-ashburn-1`` cost us an hour of debugging
# when Stelios's tenancy was actually in ``us-chicago-1``. Always
# prompt; never assume.
ORACLE_REG=$(prompt_for ORACLE_REGION "Oracle home region (e.g. us-chicago-1, us-ashburn-1)")

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
    echo "ORACLE_REGION=$ORACLE_REG"
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
printf "    ORACLE_REGION               value=%s\n" "${ORACLE_REG:-(empty)}"
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
