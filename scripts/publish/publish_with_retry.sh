#!/usr/bin/env bash
# scripts/publish/publish_with_retry.sh — paced + auto-retry orchestrator
# for the prod PyPI upload.
#
# Why this exists: the bare publish.sh hammered PyPI's burst-protection
# limit and got 429-throttled after ~4 packages. This wrapper:
#   - reads the in-repo package list (via _pypi_helpers.py),
#   - builds + uploads ONE package at a time,
#   - sleeps PACE seconds between packages,
#   - on a 429 from twine: parses ``Retry-After`` (or defaults to
#     ``RETRY_AFTER_DEFAULT``), sleeps, retries the same package up to
#     ``MAX_RETRIES`` times,
#   - tracks success in a state file under ``dist/publish/.state.txt``
#     so a re-run skips already-uploaded packages,
#   - logs to /tmp/prod_orchestrator.log so an outside watcher can
#     stream progress.
#
# Defaults: PACE=45, RETRY_AFTER_DEFAULT=600 (10 min), MAX_RETRIES=3.
# At 45s pace, 33 packages take ~25 min total (assuming no retries).
# With one rate-limit retry batch in the middle, ~45 min worst case.
#
# Auto-sources tokens from ~/.vlabs-secrets/pypi-tokens.env if present.
#
# Usage:
#   bash scripts/publish/publish_with_retry.sh --prod
#   bash scripts/publish/publish_with_retry.sh --prod --pace 60
#   bash scripts/publish/publish_with_retry.sh --prod --only verifiable-labs-envs
#   bash scripts/publish/publish_with_retry.sh --test --pace 30

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_ROOT="${REPO_ROOT}/dist/publish"
STATE_FILE="${DIST_ROOT}/.state.txt"
LOG_FILE="${ORCH_LOG:-/tmp/prod_orchestrator.log}"

# ── token source ─────────────────────────────────────────────────

_TOKENS_FILE="${HOME}/.vlabs-secrets/pypi-tokens.env"
if [ -f "$_TOKENS_FILE" ]; then
    if [ -z "${PYPI_API_TOKEN:-}" ] || [ -z "${TEST_PYPI_API_TOKEN:-}" ]; then
        # shellcheck disable=SC1090
        set -a; . "$_TOKENS_FILE"; set +a
    fi
fi

# ── arg parsing ──────────────────────────────────────────────────

MODE=""           # test | prod
ONLY=""
PACE=45           # seconds between successful uploads
RETRY_AFTER_DEFAULT=600  # seconds to wait on 429 with no Retry-After
MAX_RETRIES=3

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prod)            MODE="prod"; shift ;;
        --test)            MODE="test"; shift ;;
        --pace)            PACE="$2"; shift 2 ;;
        --retry-after)     RETRY_AFTER_DEFAULT="$2"; shift 2 ;;
        --max-retries)     MAX_RETRIES="$2"; shift 2 ;;
        --only)            ONLY="$2"; shift 2 ;;
        --reset)           rm -f "$STATE_FILE"; echo "state reset"; shift ;;
        -h|--help)         sed -n '1,30p' "$0"; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: pick --test or --prod" >&2
    exit 2
fi

if [ "$MODE" = "prod" ] && [ -z "${PYPI_API_TOKEN:-}" ]; then
    echo "ERROR: PYPI_API_TOKEN not set" >&2
    exit 2
fi
if [ "$MODE" = "test" ] && [ -z "${TEST_PYPI_API_TOKEN:-}" ]; then
    echo "ERROR: TEST_PYPI_API_TOKEN not set" >&2
    exit 2
fi

UPLOAD_URL="https://upload.pypi.org/legacy/"
[ "$MODE" = "test" ] && UPLOAD_URL="https://test.pypi.org/legacy/"

TOKEN="${PYPI_API_TOKEN:-}"
[ "$MODE" = "test" ] && TOKEN="${TEST_PYPI_API_TOKEN:-}"

mkdir -p "$DIST_ROOT"
touch "$STATE_FILE"

# ── helpers ──────────────────────────────────────────────────────

log() {
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] $*" | tee -a "$LOG_FILE"
}

state_has() {
    grep -qx "$1" "$STATE_FILE" 2>/dev/null
}

state_add() {
    echo "$1" >> "$STATE_FILE"
}

# Parse the Retry-After value out of a twine error log. PyPI returns
# the number of seconds in the header; twine echoes it inline on 429.
parse_retry_after() {
    local log_path="$1"
    grep -i "retry.after" "$log_path" 2>/dev/null \
        | head -1 \
        | grep -oE '[0-9]+' \
        | head -1
}

# Try uploading a single package's dist directory. Returns:
#   0 — success
#   1 — 429 rate limited (caller should wait + retry)
#   2 — non-retryable error (file already exists / metadata reject /
#       400 bad request / network)
upload_one() {
    local name="$1"
    local pkg_dist="${DIST_ROOT}/${name}"
    local twine_log="${pkg_dist}/twine.${MODE}.log"

    # twine wants TWINE_PASSWORD on the env, never on the cmdline.
    TWINE_USERNAME=__token__ \
    TWINE_PASSWORD="$TOKEN" \
    TWINE_NON_INTERACTIVE=1 \
        python3 -m twine upload \
            --repository-url "$UPLOAD_URL" \
            --skip-existing \
            "${pkg_dist}"/*.whl "${pkg_dist}"/*.tar.gz \
            > "$twine_log" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ]; then
        return 0
    fi

    # Strip ANSI colour for grep matches.
    local clean
    clean=$(sed 's/\x1b\[[0-9;]*m//g' "$twine_log")
    if echo "$clean" | grep -q "429 Too Many Requests"; then
        return 1
    fi
    return 2
}

build_one() {
    local name="$1"
    local pkg_dir="$2"
    local pkg_dist="${DIST_ROOT}/${name}"

    rm -rf "$pkg_dist"
    mkdir -p "$pkg_dist"
    if (cd "$pkg_dir" && python3 -m build --outdir "$pkg_dist" \
            > "${pkg_dist}/build.log" 2>&1); then
        return 0
    fi
    return 1
}

# ── enumerate packages ───────────────────────────────────────────

readarray -t PACKAGES < <(python3 - <<'PYEOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "_pypi_helpers",
    Path("scripts/publish/_pypi_helpers.py"),
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
for p in m.discover_packages():
    print(f"{p.name}|{p.version}|{p.package_dir}")
PYEOF
)

if [ "${#PACKAGES[@]}" -eq 0 ]; then
    log "ERROR: no packages discovered."
    exit 1
fi

# ── filter ───────────────────────────────────────────────────────

if [ -n "$ONLY" ]; then
    SELECTED=()
    for line in "${PACKAGES[@]}"; do
        IFS='|' read -r name _ _ <<< "$line"
        if [ "$name" = "$ONLY" ]; then
            SELECTED+=("$line")
        fi
    done
    PACKAGES=("${SELECTED[@]}")
fi

# ── main loop ────────────────────────────────────────────────────

log "=== publish_with_retry  mode=$MODE  pace=${PACE}s  selected=${#PACKAGES[@]} ==="
log "    upload_url=$UPLOAD_URL"
log "    log_file=$LOG_FILE"
log "    state_file=$STATE_FILE"
log "    already-done=$(wc -l < "$STATE_FILE" 2>/dev/null || echo 0)"

DONE=0
FAILED=0
SKIPPED=0

for line in "${PACKAGES[@]}"; do
    IFS='|' read -r name version pkg_dir <<< "$line"
    state_key="${MODE}:${name}=${version}"

    if state_has "$state_key"; then
        SKIPPED=$((SKIPPED + 1))
        log "  · ${name} ${version}  [skip — already in state file]"
        continue
    fi

    log "  · ${name} ${version}  [building...]"
    if ! build_one "$name" "$pkg_dir"; then
        log "    ⨯ BUILD FAILED — see ${DIST_ROOT}/${name}/build.log"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Upload with retry-on-429.
    rc=2
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        upload_one "$name"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            log "    ✓ uploaded (attempt $attempt)"
            break
        fi
        if [ "$rc" -eq 1 ]; then
            # 429 — parse Retry-After or default.
            ra=$(parse_retry_after "${DIST_ROOT}/${name}/twine.${MODE}.log")
            if [ -z "$ra" ]; then
                ra="$RETRY_AFTER_DEFAULT"
            fi
            log "    ⏳ 429 (attempt ${attempt}/${MAX_RETRIES}) — sleeping ${ra}s"
            sleep "$ra"
            continue
        fi
        # rc == 2 — non-retryable.
        log "    ⨯ non-retryable error — see ${DIST_ROOT}/${name}/twine.${MODE}.log"
        FAILED=$((FAILED + 1))
        break
    done

    if [ "$rc" -eq 0 ]; then
        state_add "$state_key"
        DONE=$((DONE + 1))
    elif [ "$rc" -eq 1 ]; then
        log "    ⨯ exhausted retries on 429 — moving on"
        FAILED=$((FAILED + 1))
    fi

    # Always pace between packages, even after a fail (twine rate is
    # per-IP, not per-success).
    log "    sleeping ${PACE}s before next package..."
    sleep "$PACE"
done

log "=== summary  done=$DONE  skipped=$SKIPPED  failed=$FAILED ==="
exit 0
