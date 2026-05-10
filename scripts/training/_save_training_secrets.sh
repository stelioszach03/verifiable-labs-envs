#!/usr/bin/env bash
# scripts/training/_save_training_secrets.sh — silent-prompt persister
# for the 3 training-pipeline secrets (Tier 2 + 3 + 4 work):
#
#   OPENROUTER_API_KEY    — frontier-LLM judgments for T2.6 / T2.7 +
#                            dataset prep (T4.1 / T4.2)
#   HF_TOKEN              — HuggingFace Hub auth (write scope) for
#                            RewardBench / ProcessBench downloads +
#                            dataset uploads
#   WANDB_API_KEY         — Weights & Biases run tracking (optional)
#
# Mirrors the deploy + publish savers (silent prompts, chmod 600,
# atomic write, history scrubbed). Saves to:
#
#   ~/.vlabs-secrets/training-secrets.env
#
# Subsequent training scripts auto-source the file when their env vars
# aren't already set. The repo's preflight + training tooling will be
# updated in a follow-up commit to honour this auto-source path.
#
# Use:
#   bash scripts/training/_save_training_secrets.sh
#
# To rotate / forget:
#   bash scripts/training/_save_training_secrets.sh           # overwrite
#   rm -f ~/.vlabs-secrets/training-secrets.env               # forget

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Run this with bash: bash scripts/training/_save_training_secrets.sh" >&2
    exit 1
fi

SECRETS_DIR="${HOME}/.vlabs-secrets"
TARGET="${SECRETS_DIR}/training-secrets.env"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# Atomic write via mktemp; trap cleans up if the user Ctrl+Cs mid-prompt.
TMP="$(mktemp -p "$SECRETS_DIR" .training-secrets.tmp.XXXXXX)"
chmod 600 "$TMP"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

cat <<EOF
===========================================================
Training secrets persister — Verifiable Labs

Prompts SILENTLY for the 3 training-pipeline secrets and writes
them to:

  $TARGET   (chmod 600)

Already-set keys (e.g. left over from a previous save) are
preserved unless you type a new value. Press Enter on empty
input to KEEP the existing value (or skip if no existing).

Press Ctrl+C to abort.
===========================================================
EOF

# Load existing values so blank input keeps them.
existing_openrouter=""
existing_hf=""
existing_wandb=""
if [ -f "$TARGET" ]; then
    existing_openrouter=$(awk -F= '$1=="OPENROUTER_API_KEY" {sub("^OPENROUTER_API_KEY=",""); print; exit}' "$TARGET")
    existing_hf=$(awk -F= '$1=="HF_TOKEN" {sub("^HF_TOKEN=",""); print; exit}' "$TARGET")
    existing_wandb=$(awk -F= '$1=="WANDB_API_KEY" {sub("^WANDB_API_KEY=",""); print; exit}' "$TARGET")
fi

prompt_one() {
    local name="$1"
    local format_hint="$2"
    local existing="$3"
    local _val
    if [ -n "$existing" ]; then
        echo
        echo "  ${name}:  (existing length=${#existing}; press Enter to keep)"
    else
        echo
        echo "  ${name}:  ${format_hint}"
    fi
    # shellcheck disable=SC2162
    read -srp "  Enter value (input hidden): " _val
    echo
    if [ -z "$_val" ]; then
        if [ -n "$existing" ]; then
            printf '%s=%s\n' "$name" "$existing" >> "$TMP"
            echo "  → kept existing"
        else
            echo "  → skipped (file will not contain $name)"
        fi
    else
        printf '%s=%s\n' "$name" "$_val" >> "$TMP"
        echo "  → saved (new length=${#_val})"
    fi
    unset _val
    history -d "$((HISTCMD - 1))" 2>/dev/null || true
}

# Header in the target file.
printf '# Saved %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP"

prompt_one "OPENROUTER_API_KEY" \
    "(sk-or-... — openrouter.ai/keys)" "$existing_openrouter"

prompt_one "HF_TOKEN" \
    "(hf_... — huggingface.co/settings/tokens; pick WRITE scope so dataset uploads work)" \
    "$existing_hf"

prompt_one "WANDB_API_KEY" \
    "(40-char hex — wandb.ai/authorize; press Enter to skip if you don't use W&B)" \
    "$existing_wandb"

mv "$TMP" "$TARGET"
chmod 600 "$TARGET"
trap - EXIT

# Final summary — just key names that landed; NEVER the values.
echo
echo "  Saved keys at $TARGET:"
awk -F= '/^[A-Z]/ {printf "    - %s (length=%d)\n", $1, length(substr($0, length($1)+2))}' "$TARGET"

cat <<EOF

===========================================================
Done. Subsequent runs auto-source via:

  set -a; . ~/.vlabs-secrets/training-secrets.env; set +a

Pair with the existing pypi-tokens.env if both are needed:

  set -a
  . ~/.vlabs-secrets/training-secrets.env
  . ~/.vlabs-secrets/pypi-tokens.env
  set +a
===========================================================
EOF
