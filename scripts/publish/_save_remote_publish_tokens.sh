#!/usr/bin/env bash
# scripts/publish/_save_remote_publish_tokens.sh — persist the
# GitHub PAT (+ optional Prime Intellect API key) to a chmod-600 file
# under ~/.vlabs-secrets/ so the push helper + (later) the Prime
# Intellect Hub publisher can read them without re-prompting.
#
# Mirrors scripts/publish/_save_pypi_tokens.sh and
# scripts/training/_save_training_secrets.sh:
#
#   - each prompt is silent (``read -srp``) — the values never echo
#     to the terminal AND never land in shell history;
#   - file lives at $HOME/.vlabs-secrets/remote-publish-tokens.env
#     (outside the repo tree, so git never sees it);
#   - chmod 600 on the file + chmod 700 on the directory;
#   - re-runs are idempotent: pressing Enter at a prompt keeps the
#     previously-saved value rather than wiping it.
#
# What you need before running:
#
#   GITHUB_TOKEN  — a Personal Access Token (Fine-grained or Classic).
#                   Fine-grained is preferred; minimum scopes:
#                     - "Contents: read and write"
#                     - "Metadata: read"
#                   Generated at https://github.com/settings/tokens
#                   Used by push_to_github.sh for HTTPS auth.
#
#   PRIME_INTELLECT_API_KEY  — (optional, save-for-later) The hub key
#                   from https://hub.primeintellect.ai/settings.
#                   Not used today; saved so a future env-publish
#                   script can pick it up.
#
# Usage:
#   bash scripts/publish/_save_remote_publish_tokens.sh
#
# To rotate / forget:
#   rm -f ~/.vlabs-secrets/remote-publish-tokens.env

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "ERROR: must be run via bash (not sh)." >&2
    exit 2
fi

SECRETS_DIR="$HOME/.vlabs-secrets"
SECRETS_FILE="$SECRETS_DIR/remote-publish-tokens.env"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# ── load existing values (if any) so a partial re-run can keep what
#    you already saved.
EXISTING_GITHUB_TOKEN=""
EXISTING_PRIME_INTELLECT_API_KEY=""
if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$SECRETS_FILE"; set +a
    EXISTING_GITHUB_TOKEN="${GITHUB_TOKEN:-}"
    EXISTING_PRIME_INTELLECT_API_KEY="${PRIME_INTELLECT_API_KEY:-}"
fi

echo "Verifiable Labs — remote-publish token persister"
echo ""
echo "Each prompt is silent (you'll see a cursor but no keystroke echo)."
echo "Press Enter without typing to KEEP the previously-saved value."
echo ""

# ── prompt 1: GitHub PAT ───────────────────────────────────────────
if [ -n "$EXISTING_GITHUB_TOKEN" ]; then
    label="GitHub PAT (repo scope) [existing length=${#EXISTING_GITHUB_TOKEN}]"
else
    label="GitHub PAT (repo scope)"
fi
read -srp "  ${label} > " NEW_GITHUB_TOKEN || NEW_GITHUB_TOKEN=""
echo

# ── prompt 2: Prime Intellect ──────────────────────────────────────
if [ -n "$EXISTING_PRIME_INTELLECT_API_KEY" ]; then
    label="Prime Intellect API key (optional) [existing length=${#EXISTING_PRIME_INTELLECT_API_KEY}]"
else
    label="Prime Intellect API key (optional, hub.primeintellect.ai)"
fi
read -srp "  ${label} > " NEW_PRIME_INTELLECT_API_KEY || NEW_PRIME_INTELLECT_API_KEY=""
echo

# ── resolve: prefer new value when supplied, fall back to existing ─
FINAL_GITHUB="${NEW_GITHUB_TOKEN:-$EXISTING_GITHUB_TOKEN}"
FINAL_PRIME="${NEW_PRIME_INTELLECT_API_KEY:-$EXISTING_PRIME_INTELLECT_API_KEY}"

if [ -z "$FINAL_GITHUB" ]; then
    echo "" >&2
    echo "ERROR: GitHub PAT is required. Run again and paste it at the prompt." >&2
    exit 2
fi

# ── write atomically with chmod 600 ────────────────────────────────
TMP="$(mktemp "${SECRETS_DIR}/.remote-publish.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
{
    echo "# Remote-publish credentials — written by"
    echo "# scripts/publish/_save_remote_publish_tokens.sh"
    echo "# DO NOT commit. chmod 600 enforced below."
    echo "GITHUB_TOKEN=$FINAL_GITHUB"
    echo "PRIME_INTELLECT_API_KEY=$FINAL_PRIME"
} > "$TMP"
chmod 600 "$TMP"
mv "$TMP" "$SECRETS_FILE"
trap - EXIT

# ── confirm without echoing values ────────────────────────────────
echo ""
echo "  ✓ Saved to $SECRETS_FILE (chmod 600)"
echo ""
echo "    GITHUB_TOKEN              length=${#FINAL_GITHUB}"
echo "    PRIME_INTELLECT_API_KEY   length=${#FINAL_PRIME}"
echo ""
echo "Next step — push the 89 unpushed commits to GitHub:"
echo "    bash scripts/publish/push_to_github.sh --dry-run    # preview"
echo "    bash scripts/publish/push_to_github.sh --yes        # actually push"
