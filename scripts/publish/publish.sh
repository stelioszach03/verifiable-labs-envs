#!/usr/bin/env bash
# scripts/publish/publish.sh — build + upload in-repo packages to
# PyPI / Test PyPI.
#
# Reads tokens from $PYPI_API_TOKEN / $TEST_PYPI_API_TOKEN (loaded
# via _load_pypi_secrets.sh). NEVER pastes them anywhere; passes via
# the env to twine. Skips packages whose local version is already
# uploaded (twine --skip-existing).
#
# Usage:
#   bash scripts/publish/publish.sh --list
#   bash scripts/publish/publish.sh --test --all
#   bash scripts/publish/publish.sh --test --package verifiable-labs-envs
#   bash scripts/publish/publish.sh --prod --all
#   bash scripts/publish/publish.sh --dry-run --all              # build only
#
# Defaults to --test for safety. --prod uploads to real PyPI
# (irreversible). Always run --test first.
#
# Exit codes:
#   0 — every selected package built (and uploaded if --test/--prod).
#   1 — one or more packages failed to build / upload.
#   2 — bad arguments / missing tokens for the chosen index.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_ROOT="${REPO_ROOT}/dist/publish"

# ── arg parsing ──────────────────────────────────────────────────

MODE=""           # test | prod | dry-run | list
ONLY_PACKAGE=""
SKIP_EXISTING=1   # twine --skip-existing on by default

while [ "$#" -gt 0 ]; do
    case "$1" in
        --list)            MODE="list";   shift ;;
        --test)            MODE="test";   shift ;;
        --prod)            MODE="prod";   shift ;;
        --dry-run)         MODE="dry";    shift ;;
        --package)         ONLY_PACKAGE="$2"; shift 2 ;;
        --all)             ONLY_PACKAGE=""; shift ;;
        --no-skip-existing) SKIP_EXISTING=0; shift ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: pick a mode: --list | --test | --prod | --dry-run" >&2
    exit 2
fi

# ── token check ──────────────────────────────────────────────────

if [ "$MODE" = "test" ] && [ -z "${TEST_PYPI_API_TOKEN:-}" ]; then
    echo "ERROR: TEST_PYPI_API_TOKEN not set." >&2
    echo "       Run: source scripts/publish/_load_pypi_secrets.sh --only test" >&2
    exit 2
fi
if [ "$MODE" = "prod" ] && [ -z "${PYPI_API_TOKEN:-}" ]; then
    echo "ERROR: PYPI_API_TOKEN not set." >&2
    echo "       Run: source scripts/publish/_load_pypi_secrets.sh --only prod" >&2
    exit 2
fi

# ── tooling check ────────────────────────────────────────────────

ensure_tool() {
    local mod="$1"
    if ! python3 -c "import $mod" >/dev/null 2>&1; then
        echo "INFO: installing $mod (--user --break-system-packages)..."
        pip install --user --break-system-packages --quiet "$mod" || {
            echo "ERROR: failed to install $mod" >&2
            exit 2
        }
    fi
}

if [ "$MODE" != "list" ]; then
    ensure_tool "build"
fi
if [ "$MODE" = "test" ] || [ "$MODE" = "prod" ]; then
    ensure_tool "twine"
fi
ensure_tool "httpx"

# ── list mode (delegate to list_packages.py) ─────────────────────

if [ "$MODE" = "list" ]; then
    exec python3 "${REPO_ROOT}/scripts/publish/list_packages.py" "$@"
fi

# ── enumerate target packages ────────────────────────────────────

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
    echo "ERROR: no packages discovered." >&2
    exit 1
fi

if [ -n "$ONLY_PACKAGE" ]; then
    FILTERED=()
    for line in "${PACKAGES[@]}"; do
        IFS='|' read -r name _ _ <<< "$line"
        if [ "$name" = "$ONLY_PACKAGE" ]; then
            FILTERED+=("$line")
        fi
    done
    if [ "${#FILTERED[@]}" -eq 0 ]; then
        echo "ERROR: --package $ONLY_PACKAGE not found." >&2
        exit 1
    fi
    PACKAGES=("${FILTERED[@]}")
fi

echo "=== publishing ${#PACKAGES[@]} package(s) — mode=$MODE ==="
mkdir -p "$DIST_ROOT"

FAIL=0

for line in "${PACKAGES[@]}"; do
    IFS='|' read -r name version pkg_dir <<< "$line"

    pkg_dist="${DIST_ROOT}/${name}"
    rm -rf "$pkg_dist"
    mkdir -p "$pkg_dist"

    echo
    echo "--- ${name} ${version} ---"
    echo "    src:  $pkg_dir"
    echo "    dist: $pkg_dist"

    # Build the wheel + sdist into the per-package dist dir.
    if ! (cd "$pkg_dir" && python3 -m build --outdir "$pkg_dist" \
            > "${pkg_dist}/build.log" 2>&1); then
        echo "    BUILD FAILED — see ${pkg_dist}/build.log"
        FAIL=$((FAIL + 1))
        continue
    fi

    artifacts=("${pkg_dist}"/*.whl "${pkg_dist}"/*.tar.gz)
    echo "    built: $(basename "${artifacts[0]}") + $(basename "${artifacts[1]}")"

    if [ "$MODE" = "dry" ]; then
        echo "    [dry-run] not uploading"
        continue
    fi

    # twine upload via env-passed token (no plaintext on the cmdline).
    if [ "$MODE" = "test" ]; then
        TWINE_USERNAME=__token__ \
        TWINE_PASSWORD="$TEST_PYPI_API_TOKEN" \
        TWINE_NON_INTERACTIVE=1 \
            python3 -m twine upload \
                --repository-url https://test.pypi.org/legacy/ \
                $([ "$SKIP_EXISTING" = "1" ] && echo --skip-existing) \
                "${pkg_dist}"/*.whl "${pkg_dist}"/*.tar.gz \
                > "${pkg_dist}/twine.log" 2>&1 || {
            echo "    TWINE UPLOAD FAILED — see ${pkg_dist}/twine.log"
            FAIL=$((FAIL + 1))
            continue
        }
        echo "    ✓ uploaded to test.pypi"
    else  # prod
        TWINE_USERNAME=__token__ \
        TWINE_PASSWORD="$PYPI_API_TOKEN" \
        TWINE_NON_INTERACTIVE=1 \
            python3 -m twine upload \
                $([ "$SKIP_EXISTING" = "1" ] && echo --skip-existing) \
                "${pkg_dist}"/*.whl "${pkg_dist}"/*.tar.gz \
                > "${pkg_dist}/twine.log" 2>&1 || {
            echo "    TWINE UPLOAD FAILED — see ${pkg_dist}/twine.log"
            FAIL=$((FAIL + 1))
            continue
        }
        echo "    ✓ uploaded to pypi.org"
    fi
done

echo
echo "=== summary ==="
echo "    selected: ${#PACKAGES[@]}"
echo "    failed:   $FAIL"
echo "    artifacts under: $DIST_ROOT"
echo

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
