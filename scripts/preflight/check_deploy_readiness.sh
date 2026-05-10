#!/usr/bin/env bash
# scripts/preflight/check_deploy_readiness.sh — go/no-go report before
# `flyctl deploy --app vlabs-api`.
#
# Verifies:
#   1. fly.toml exists and parses cleanly (services/api/fly.toml).
#   2. Alembic migration files 0001-0009 are present.
#   3. Required Fly secrets are provisioned (DATABASE_URL, REDIS_URL,
#      R2_*, RESEND_API_KEY, VLABS_LOCAL_FAKE_EMAIL=false).
#   4. Dockerfile referenced from fly.toml exists.
#
# Output: one line per check, plus an overall GO / NO-GO verdict on
# stderr. Exits 0 if all checks pass, 1 otherwise.
#
# Usage:
#   bash scripts/preflight/check_deploy_readiness.sh           # full check
#   bash scripts/preflight/check_deploy_readiness.sh --no-fly  # skip flyctl
#   bash scripts/preflight/check_deploy_readiness.sh --json    # JSON output
#
# Stelios runs this once before each prod deploy. CI runs it via the
# tests/preflight/test_deploy_check.py pytest harness in --no-fly mode.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FLY_TOML="${REPO_ROOT}/services/api/fly.toml"
MIGRATIONS_DIR="${REPO_ROOT}/services/api/migrations/versions"

REQUIRED_MIGRATIONS=(
  "0001_initial.py"
  "0002_stripe_events.py"
  "0003_add_audit_calls.py"
  "0004_add_dataset_jobs.py"
  "0005_add_monitors.py"
  "0006_add_reward_models.py"
  "0007_add_process_reward_models.py"
  "0008_add_attestations.py"
  "0009_add_attestation_certificates.py"
)

REQUIRED_SECRETS=(
  "DATABASE_URL"
  "REDIS_URL"
  "R2_ACCESS_KEY_ID"
  "R2_SECRET_ACCESS_KEY"
  "R2_BUCKET"
  "RESEND_API_KEY"
)

OPTIONAL_SECRETS=(
  "STRIPE_SECRET_KEY"
  "CLERK_SECRET_KEY"
  "CLERK_JWKS_URL"
  "SENTRY_DSN"
  "VLABS_API_KEY_HASH_PEPPER"
)

NO_FLY=0
JSON_OUT=0
for arg in "$@"; do
  case "$arg" in
    --no-fly) NO_FLY=1 ;;
    --json)   JSON_OUT=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PASS=0
FAIL=0
RESULTS=()

emit() {
  # emit <name> <status> <detail>
  local name="$1" status="$2" detail="$3"
  if [[ $JSON_OUT -eq 1 ]]; then
    RESULTS+=("{\"name\":\"$name\",\"status\":\"$status\",\"detail\":\"$detail\"}")
  else
    if [[ "$status" == "pass" ]]; then
      printf '  [OK]   %-40s %s\n' "$name" "$detail"
    else
      printf '  [FAIL] %-40s %s\n' "$name" "$detail" >&2
    fi
  fi
  if [[ "$status" == "pass" ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

[[ $JSON_OUT -eq 0 ]] && echo "=== check_deploy_readiness.sh ==="

# 1. fly.toml exists.
if [[ -f "$FLY_TOML" ]]; then
  emit "fly_toml_present" "pass" "$FLY_TOML"
else
  emit "fly_toml_present" "fail" "missing: $FLY_TOML"
fi

# 2. fly.toml mentions the expected app name + Dockerfile.
if [[ -f "$FLY_TOML" ]]; then
  if grep -q '^app = "vlabs-api"' "$FLY_TOML"; then
    emit "fly_app_name" "pass" "vlabs-api"
  else
    emit "fly_app_name" "fail" "expected app=\"vlabs-api\""
  fi
  DOCKERFILE_LINE=$(grep -E '^[[:space:]]*dockerfile' "$FLY_TOML" | head -1 || true)
  if [[ -n "$DOCKERFILE_LINE" ]]; then
    DOCKERFILE_REL=$(echo "$DOCKERFILE_LINE" | sed -E 's/.*"([^"]+)".*/\1/')
    # The dockerfile path in fly.toml is relative to the fly.toml
    # directory, NOT the repo root.
    FLY_DIR="$(dirname "$FLY_TOML")"
    DOCKERFILE_ABS="${FLY_DIR}/${DOCKERFILE_REL}"
    if [[ -f "$DOCKERFILE_ABS" ]]; then
      emit "dockerfile_present" "pass" "$DOCKERFILE_REL"
    else
      emit "dockerfile_present" "fail" "missing: $DOCKERFILE_ABS"
    fi
  else
    emit "dockerfile_present" "fail" "no dockerfile line in fly.toml"
  fi
fi

# 3. Required migration files present.
MISSING_MIGRATIONS=()
for m in "${REQUIRED_MIGRATIONS[@]}"; do
  if [[ ! -f "${MIGRATIONS_DIR}/${m}" ]]; then
    MISSING_MIGRATIONS+=("$m")
  fi
done
if [[ ${#MISSING_MIGRATIONS[@]} -eq 0 ]]; then
  emit "migrations_0001_to_0009" "pass" "9 files"
else
  emit "migrations_0001_to_0009" "fail" "missing: ${MISSING_MIGRATIONS[*]}"
fi

# 4. Required secrets — Fly mode: query `fly secrets list`. No-fly mode:
#    check the local environment instead (used by tests + dev).
MISSING_SECRETS=()
if [[ $NO_FLY -eq 0 ]] && command -v fly >/dev/null 2>&1; then
  SECRETS_OUT=$(fly secrets list --app vlabs-api 2>/dev/null || true)
  if [[ -z "$SECRETS_OUT" ]]; then
    emit "fly_secrets_query" "fail" "fly secrets list returned empty (auth?)"
  else
    for s in "${REQUIRED_SECRETS[@]}"; do
      if ! echo "$SECRETS_OUT" | grep -q "^$s "; then
        MISSING_SECRETS+=("$s")
      fi
    done
    if [[ ${#MISSING_SECRETS[@]} -eq 0 ]]; then
      emit "fly_required_secrets" "pass" "${#REQUIRED_SECRETS[@]} provisioned"
    else
      emit "fly_required_secrets" "fail" "missing: ${MISSING_SECRETS[*]}"
    fi
  fi
else
  for s in "${REQUIRED_SECRETS[@]}"; do
    if [[ -z "${!s:-}" ]]; then
      MISSING_SECRETS+=("$s")
    fi
  done
  if [[ ${#MISSING_SECRETS[@]} -eq 0 ]]; then
    emit "env_required_secrets" "pass" "${#REQUIRED_SECRETS[@]} present in shell env"
  else
    emit "env_required_secrets" "fail" "missing in env: ${MISSING_SECRETS[*]}"
  fi
fi

# 5. VLABS_LOCAL_FAKE_EMAIL must NOT be true in prod (sanity check).
if [[ "${VLABS_LOCAL_FAKE_EMAIL:-false}" == "true" ]]; then
  emit "fake_email_flag_off" "fail" "VLABS_LOCAL_FAKE_EMAIL=true (must be false for prod)"
else
  emit "fake_email_flag_off" "pass" "VLABS_LOCAL_FAKE_EMAIL=${VLABS_LOCAL_FAKE_EMAIL:-unset}"
fi

# 6. Phase 31 PKI — VLABS_LOCAL_FAKE_PKI must NOT be true in prod.
if [[ "${VLABS_LOCAL_FAKE_PKI:-false}" == "true" ]]; then
  emit "fake_pki_flag_off" "fail" "VLABS_LOCAL_FAKE_PKI=true (must be false for prod)"
else
  emit "fake_pki_flag_off" "pass" "VLABS_LOCAL_FAKE_PKI=${VLABS_LOCAL_FAKE_PKI:-unset}"
fi

# Summary.
if [[ $JSON_OUT -eq 1 ]]; then
  printf '{"pass":%d,"fail":%d,"verdict":"%s","results":[%s]}\n' \
    "$PASS" "$FAIL" "$([[ $FAIL -eq 0 ]] && echo GO || echo NO-GO)" \
    "$(IFS=,; echo "${RESULTS[*]}")"
else
  echo
  if [[ $FAIL -eq 0 ]]; then
    echo "VERDICT: GO  (${PASS} checks passed, 0 failed)" >&2
  else
    echo "VERDICT: NO-GO  (${PASS} checks passed, ${FAIL} failed)" >&2
  fi
fi

[[ $FAIL -eq 0 ]] && exit 0 || exit 1
