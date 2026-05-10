#!/usr/bin/env bash
# scripts/publish/push_to_github.sh — rewrite the local unpushed
# commits to a single canonical author identity + push to GitHub.
#
# Why this exists
# ───────────────
#
# Every commit made through Claude Code carries a
#
#     Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
#
# trailer per the assistant's default git-commit template. Pushing
# those as-is would credit Claude as a co-author on every overnight
# commit in the GitHub profile, which the maintainer doesn't want
# (the SDK is single-author for funding / PhD application context).
#
# Two name variants ("Stelios" vs "Stelios Zacharioudakis") exist in
# the local unpushed range from older sessions where git config drifted
# between machines. Both use the same email
# (sdi2200243@di.uoa.gr) so this script normalises ALL unpushed
# commits to:
#
#     Stelios Zacharioudakis <sdi2200243@di.uoa.gr>
#
# What it does (in order)
# ───────────────────────
#
#   1. Load GITHUB_TOKEN from ~/.vlabs-secrets/remote-publish-tokens.env
#      (created by _save_remote_publish_tokens.sh).
#   2. ``git fetch origin`` to refresh remote-tracking refs.
#   3. Compute COMMITS = origin/main..HEAD — only LOCAL unpushed
#      commits get rewritten; anything already on GitHub is left alone.
#   4. Refuse to run if the working tree is dirty (filter-branch
#      cannot operate on a dirty WT cleanly).
#   5. ``git filter-branch`` over the range:
#        - rewrite author + committer to the canonical identity;
#        - strip ``Co-Authored-By: …Claude…`` / ``…Anthropic…`` lines
#          from each commit message.
#   6. Verify the rewrite (author set is exactly the canonical one;
#      no Co-Authored-By trailers remain).
#   7. Push HEAD → origin/main via HTTPS-with-PAT — the PAT is
#      embedded in the URL only for this single push (not written to
#      .git/config).
#
# Safety
# ──────
#   - Only LOCAL unpushed commits are touched. Anything already on
#     origin/main is left alone — the published history stays intact.
#   - filter-branch creates a backup ref under refs/original/. Once
#     you've confirmed the push, clean it up with:
#         git update-ref -d refs/original/refs/heads/main
#   - ``--dry-run`` shows the impact (commit count, author mix,
#     trailers) WITHOUT modifying anything.
#   - ``--yes`` skips the interactive confirmation prompt — required
#     for non-interactive (assistant / CI) execution.
#
# Usage
# ─────
#
#   bash scripts/publish/push_to_github.sh --dry-run
#   bash scripts/publish/push_to_github.sh --yes
#   bash scripts/publish/push_to_github.sh                # interactive
#
# Idempotent: a re-run after a successful push is a no-op (origin/main
# == HEAD, so COMMITS = 0).

set -euo pipefail

# ── config — change if you fork ────────────────────────────────────
REPO="stelioszach03/verifiable-labs-envs"
AUTHOR_NAME="Stelios Zacharioudakis"
AUTHOR_EMAIL="sdi2200243@di.uoa.gr"
REMOTE_BRANCH="main"
TOKENS_FILE="$HOME/.vlabs-secrets/remote-publish-tokens.env"

# ── arg parsing ────────────────────────────────────────────────────
DRY_RUN=0
ASSUME_YES=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --yes|-y)        ASSUME_YES=1; shift ;;
        -h|--help)
            sed -n '1,60p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

# ── load token ─────────────────────────────────────────────────────
if [ ! -f "$TOKENS_FILE" ]; then
    echo "ERROR: $TOKENS_FILE not found." >&2
    echo "  Run first: bash scripts/publish/_save_remote_publish_tokens.sh" >&2
    exit 2
fi
# shellcheck disable=SC1090
set -a; source "$TOKENS_FILE"; set +a
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN missing in $TOKENS_FILE." >&2
    exit 2
fi

# ── move to repo root ──────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ── 1. fetch ───────────────────────────────────────────────────────
echo "  → git fetch origin"
git fetch --quiet origin

# ── 2. compute range ───────────────────────────────────────────────
UPSTREAM="origin/$REMOTE_BRANCH"
if ! git rev-parse --verify "$UPSTREAM" >/dev/null 2>&1; then
    echo "ERROR: $UPSTREAM doesn't exist on origin." >&2
    exit 2
fi

N_COMMITS=$(git rev-list --count "$UPSTREAM..HEAD")
if [ "$N_COMMITS" -eq 0 ]; then
    echo "  ✓ nothing to push: HEAD == $UPSTREAM"
    exit 0
fi

echo ""
echo "  → $N_COMMITS unpushed commit(s) on top of $UPSTREAM"
echo ""
echo "  authors that WILL be rewritten:"
git log --format='    %an <%ae>' "$UPSTREAM..HEAD" | sort | uniq -c | sort -rn | sed 's/^/  /'
echo ""
echo "  Co-Authored-By trailers that WILL be stripped:"
TRAILERS=$(git log --format='%B' "$UPSTREAM..HEAD" | grep -i 'Co-Authored-By' | sort -u || true)
if [ -z "$TRAILERS" ]; then
    echo "    (none — nothing to strip)"
else
    echo "$TRAILERS" | sed 's/^/    /'
fi

# ── 3. dirty WT detection — auto-stash with restore on exit ───────
#
# filter-branch enforces a clean working tree. The Phase 18 v5
# contract deliberately keeps PHASE_18_v5_PLAN.md modified + a known
# set of untracked files in the tree, so we can't ask the maintainer
# to manually stash. Instead: stash silently (incl. untracked, with
# a recognisable label), run filter-branch, restore on exit.
NEEDS_STASH=0
STASH_LABEL="push_to_github auto-stash $(date -u +%Y%m%dT%H%M%SZ)"
if [ -n "$(git status --porcelain)" ]; then
    NEEDS_STASH=1
    echo ""
    echo "  → working tree is dirty; auto-stashing (label: $STASH_LABEL)"
fi

# ── 4. dry-run early-exit ──────────────────────────────────────────
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "  --dry-run set; not rewriting or pushing."
    if [ "$NEEDS_STASH" -eq 1 ]; then
        echo "  (would auto-stash these entries before rewriting:)"
        git status --short | sed 's/^/    /'
    fi
    exit 0
fi

restore_stash() {
    if [ "$NEEDS_STASH" -eq 1 ]; then
        echo ""
        echo "  → restoring auto-stashed working tree..."
        # ``git stash pop`` returns non-zero on merge conflict but we
        # don't want the trap to swallow it silently.
        if ! git stash pop --quiet; then
            echo "  ⚠ stash pop reported conflicts. Inspect:" >&2
            echo "      git stash list" >&2
            echo "      git status" >&2
        fi
    fi
}

# ── 5. confirm ─────────────────────────────────────────────────────
if [ "$ASSUME_YES" -ne 1 ]; then
    echo ""
    read -p "  Proceed with rewrite + push? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "  aborted."; exit 0 ;;
    esac
fi

# ── 6. rewrite ─────────────────────────────────────────────────────
echo ""

# Stash NOW (after the confirm prompt) so a user who answers "no"
# doesn't end up with their WT in a stash.
if [ "$NEEDS_STASH" -eq 1 ]; then
    git stash push --include-untracked --quiet -m "$STASH_LABEL"
    trap restore_stash EXIT
fi

echo "  → rewriting $N_COMMITS commits via git filter-branch..."

# Wipe any stale backup ref so filter-branch doesn't refuse to run.
rm -rf .git/refs/original 2>/dev/null || true

# Use the canonical identity for both author + committer. Strip any
# Co-Authored-By trailer whose value contains Claude or Anthropic.
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
    --env-filter "
        export GIT_AUTHOR_NAME='${AUTHOR_NAME}'
        export GIT_AUTHOR_EMAIL='${AUTHOR_EMAIL}'
        export GIT_COMMITTER_NAME='${AUTHOR_NAME}'
        export GIT_COMMITTER_EMAIL='${AUTHOR_EMAIL}'
    " \
    --msg-filter '
        grep -v -i -E "^Co-Authored-By:.*(Claude|Anthropic)" || true
    ' \
    "$UPSTREAM..HEAD"

# ── 7. verify ──────────────────────────────────────────────────────
echo ""
echo "  ✓ rewrite complete. Verifying..."

REMAINING_AUTHORS=$(git log --format='%an <%ae>' "$UPSTREAM..HEAD" | sort -u)
EXPECTED="${AUTHOR_NAME} <${AUTHOR_EMAIL}>"
if [ "$REMAINING_AUTHORS" != "$EXPECTED" ]; then
    echo "ERROR: unexpected author(s) remain after rewrite:" >&2
    echo "$REMAINING_AUTHORS" >&2
    echo "expected: $EXPECTED" >&2
    exit 1
fi
echo "    authors: $REMAINING_AUTHORS"

if git log --format='%B' "$UPSTREAM..HEAD" | grep -qi 'Co-Authored-By'; then
    echo "ERROR: Co-Authored-By trailers still present after rewrite:" >&2
    git log --format='%B' "$UPSTREAM..HEAD" | grep -i 'Co-Authored-By' | sort -u >&2
    exit 1
fi
echo "    no Co-Authored-By trailers remain"

# ── 8. push ────────────────────────────────────────────────────────
echo ""
echo "  → pushing to origin/$REMOTE_BRANCH..."

# Push via a one-off URL with the PAT embedded; this DOESN'T touch
# .git/config (origin remote stays whatever it was).
PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git"
git push "$PUSH_URL" "HEAD:$REMOTE_BRANCH"

echo ""
echo "  ✓ pushed."
echo "  Remote: https://github.com/$REPO"
echo ""
echo "  (Optional) clean up the filter-branch backup ref:"
echo "      git update-ref -d refs/original/refs/heads/${REMOTE_BRANCH}"
