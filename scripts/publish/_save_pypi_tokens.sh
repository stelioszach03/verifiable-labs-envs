#!/usr/bin/env bash
# scripts/publish/_save_pypi_tokens.sh — persist PyPI + Test PyPI
# tokens to a chmod-600 file under ~/.vlabs-secrets/, so subsequent
# publish invocations (run from this shell or any other) can pick
# them up without re-prompting.
#
# Mirrors the existing _load_pypi_secrets.sh silent-prompt pattern,
# but writes to disk explicitly (the loader keeps things in-shell
# only). The file path is OUTSIDE the repo and gitignored at the
# home directory level by virtue of being under ~/.vlabs-secrets/
# (a directory the repo's git tree never touches).
#
# Use:
#   bash scripts/publish/_save_pypi_tokens.sh
#
# After this runs, scripts/publish/publish.sh and
# scripts/publish/yank_old_versions.py auto-source the file when
# their PYPI_API_TOKEN / TEST_PYPI_API_TOKEN env vars are unset.
#
# To rotate / forget the tokens:
#   rm -f ~/.vlabs-secrets/pypi-tokens.env

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Run this with bash: bash scripts/publish/_save_pypi_tokens.sh" >&2
    exit 1
fi

SECRETS_DIR="${HOME}/.vlabs-secrets"
TOKENS_FILE="${SECRETS_DIR}/pypi-tokens.env"

mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

# Always start from a clean tmp file so a partial run doesn't leave
# a half-written tokens file behind.
TMP="$(mktemp -p "$SECRETS_DIR" .pypi-tokens.tmp.XXXXXX)"
chmod 600 "$TMP"

cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

cat <<EOF
===========================================================
PyPI tokens saver — Verifiable Labs

This script prompts SILENTLY for your PyPI + Test PyPI API
tokens and writes them to:

  $TOKENS_FILE

(chmod 600, owner-only).

Input is HIDDEN at the terminal. Press Enter to skip a token.
Press Ctrl+C to abort.
===========================================================
EOF

read -srp "PYPI_API_TOKEN (pypi-... — blank to skip): " _prod_token
echo
read -srp "TEST_PYPI_API_TOKEN (pypi-... — blank to skip): " _test_token
echo

{
    echo "# Saved $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ -n "$_prod_token" ]; then
        printf 'PYPI_API_TOKEN=%s\n' "$_prod_token"
    fi
    if [ -n "$_test_token" ]; then
        printf 'TEST_PYPI_API_TOKEN=%s\n' "$_test_token"
    fi
} > "$TMP"
mv "$TMP" "$TOKENS_FILE"
chmod 600 "$TOKENS_FILE"
trap - EXIT

# Zero out + clear from history.
unset _prod_token _test_token
history -d "$((HISTCMD - 1))" 2>/dev/null || true

cat <<EOF

Tokens saved to $TOKENS_FILE (chmod 600).

Next:
  bash scripts/publish/publish.sh --list
  bash scripts/publish/publish.sh --test --all
  bash scripts/publish/publish.sh --prod --all

To rotate later:
  bash scripts/publish/_save_pypi_tokens.sh   # overwrite
  rm -f $TOKENS_FILE                          # forget entirely
EOF
