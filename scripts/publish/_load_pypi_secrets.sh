#!/usr/bin/env bash
# scripts/publish/_load_pypi_secrets.sh — silent-prompt loader for the
# PyPI publishing toolkit. Mirrors scripts/deploy/_load_provider_
# secrets.sh: NEVER prints tokens, NEVER paste in chat, must be sourced
# (not executed).
#
# Exports:
#   PYPI_API_TOKEN          — for pypi.org    (write-permission scope)
#   TEST_PYPI_API_TOKEN     — for test.pypi.org (write-permission scope)
#
# Both tokens look like ``pypi-AgEIcHlwaS5vcmcCJ...`` (long base64).
# Generate from:
#   https://pypi.org/manage/account/token/
#   https://test.pypi.org/manage/account/token/
#
# Use:
#   source scripts/publish/_load_pypi_secrets.sh
#
# Optional override:
#   source scripts/publish/_load_pypi_secrets.sh --only test    # only test.pypi
#   source scripts/publish/_load_pypi_secrets.sh --only prod    # only real pypi

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Run this with bash: source scripts/publish/_load_pypi_secrets.sh" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: this script must be sourced, not executed directly." >&2
    echo "Try: source scripts/publish/_load_pypi_secrets.sh" >&2
    exit 1
fi

_only_filter=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --only)
            _only_filter="$2"
            shift 2
            ;;
        *)
            echo "unknown arg: $1" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

_should_load() {
    local target="$1"
    if [ -z "$_only_filter" ]; then
        return 0
    fi
    [ "$_only_filter" = "$target" ]
}

# Optional persistent stamp dir at ~/.vlabs-secrets/. Mirrors the
# deploy provisioner. Never writes the token values; only a chmod-600
# stamp file noting which were loaded in the current session.
_secrets_dir="${HOME}/.vlabs-secrets"
mkdir -p "$_secrets_dir"
chmod 700 "$_secrets_dir"

_loaded=()

if _should_load "prod"; then
    read -srp "PYPI_API_TOKEN (pypi-... — blank to skip): " _val
    echo
    if [ -n "$_val" ]; then
        export PYPI_API_TOKEN="$_val"
        _loaded+=("PYPI_API_TOKEN")
    fi
    unset _val
    history -d "$((HISTCMD - 1))" 2>/dev/null || true
fi

if _should_load "test"; then
    read -srp "TEST_PYPI_API_TOKEN (pypi-... — blank to skip): " _val
    echo
    if [ -n "$_val" ]; then
        export TEST_PYPI_API_TOKEN="$_val"
        _loaded+=("TEST_PYPI_API_TOKEN")
    fi
    unset _val
    history -d "$((HISTCMD - 1))" 2>/dev/null || true
fi

# Drop the stamp file (key names only — NO values).
{
    echo "# pypi-secrets loaded $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for v in "${_loaded[@]}"; do
        echo "$v"
    done
} > "$_secrets_dir/last_pypi_load.txt"
chmod 600 "$_secrets_dir/last_pypi_load.txt"

if [ "${#_loaded[@]}" -gt 0 ]; then
    echo "loaded:  ${_loaded[*]}"
else
    echo "loaded:  (none — both prompts skipped)"
fi
echo "stamp:   $_secrets_dir/last_pypi_load.txt"

unset _only_filter _loaded _secrets_dir
unset -f _should_load
