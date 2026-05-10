#!/usr/bin/env bash
# scripts/preflight/_load_provider_secrets.sh — multi-provider auth
# token loader. Mirrors _load_phase28_secrets.sh: prompts silently,
# exports to the current shell, never persists to disk.
#
# Usage:
#   source scripts/preflight/_load_provider_secrets.sh           # prompt all
#   source scripts/preflight/_load_provider_secrets.sh --only HF_TOKEN,WANDB_API_KEY
#
# Stelios runs this once per shell after collecting provider tokens
# from the parallel signup process (Vultr, RunPod, DO, GCP, Azure,
# Oracle, Thunder, Modal, HF, W&B, OpenRouter). Tests skip this loader
# entirely; they synthesise fake tokens via monkeypatch.

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this with bash:  source scripts/preflight/_load_provider_secrets.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: this script must be sourced, not executed directly." >&2
  echo "Try:   source scripts/preflight/_load_provider_secrets.sh" >&2
  exit 1
fi

# All provider tokens we know about (env-var name + human label).
declare -a _provider_vars=(
  "VULTR_API_KEY"
  "RUNPOD_API_KEY"
  "DIGITALOCEAN_API_TOKEN"
  "GCP_SERVICE_ACCOUNT_JSON"
  "AZURE_CLIENT_SECRET"
  "ORACLE_CLI_AUTH_TOKEN"
  "THUNDER_COMPUTE_API_KEY"
  "MODAL_TOKEN_ID"
  "HF_TOKEN"
  "WANDB_API_KEY"
  "OPENROUTER_API_KEY"
)

# Optional --only filter: comma-separated subset of var names.
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
  local name="$1"
  if [ -z "$_only_filter" ]; then
    return 0
  fi
  case ",$_only_filter," in
    *",$name,"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Optional persistent secrets dir at ~/.vlabs-secrets/. The loader
# never writes API keys here — only a stamp file so subsequent
# preflight scripts can see which providers were configured this
# shell session without re-prompting.
_secrets_dir="${HOME}/.vlabs-secrets"
mkdir -p "$_secrets_dir"
chmod 700 "$_secrets_dir"

_loaded=()
_skipped=()

for _var in "${_provider_vars[@]}"; do
  if ! _should_load "$_var"; then
    continue
  fi
  read -srp "${_var} (blank to skip): " _value
  echo
  if [ -n "$_value" ]; then
    export "$_var=$_value"
    _loaded+=("$_var")
  else
    _skipped+=("$_var")
  fi
  unset _value
done

# Drop the (non-secret) stamp file listing which providers are now
# present in the current shell. NEVER write the actual values.
{
  echo "# Loaded $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for v in "${_loaded[@]}"; do
    echo "$v"
  done
} > "$_secrets_dir/last_load.txt"
chmod 600 "$_secrets_dir/last_load.txt"

if [ "${#_loaded[@]}" -gt 0 ]; then
  echo "loaded:  ${_loaded[*]}"
fi
if [ "${#_skipped[@]}" -gt 0 ]; then
  echo "skipped: ${_skipped[*]}"
fi
echo "stamp:   $_secrets_dir/last_load.txt"

unset _provider_vars _only_filter _loaded _skipped _var _secrets_dir
unset -f _should_load
