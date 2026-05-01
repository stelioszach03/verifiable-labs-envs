#!/bin/bash
# Deploy services/landing to Cloudflare Pages.
#
# Pre-reqs:
#   - npm install (one-time)
#   - wrangler installed: npm install -g wrangler  (or via npx)
#   - wrangler login    (browser flow; CLOUDFLARE_API_TOKEN works in CI too)
#   - .env.local present with NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY etc.
#
# Idempotent: re-running rebuilds + uploads. The Cloudflare Pages project
# itself is created on first run via `wrangler pages project create`.
#
#   ./services/landing/deploy/cloudflare-deploy.sh

set -euo pipefail

PROJECT_NAME="vlabs-landing"
HERE="$(cd "$(dirname "$0")" && pwd)"
LANDING_ROOT="$(cd "$HERE/.." && pwd)"

cd "$LANDING_ROOT"

if [ ! -f package.json ]; then
    echo "[landing-deploy] ERROR: not in a Next.js project root" >&2
    exit 1
fi

# Step 1 — install deps if needed.
if [ ! -d node_modules ]; then
    echo "[landing-deploy] installing npm deps"
    npm ci
fi

# Step 2 — Cloudflare Pages requires a static build via @cloudflare/next-on-pages.
echo "[landing-deploy] running next build"
npm run build

if ! npx --yes @cloudflare/next-on-pages --version >/dev/null 2>&1; then
    echo "[landing-deploy] adding @cloudflare/next-on-pages"
    npm install --save-dev @cloudflare/next-on-pages
fi

echo "[landing-deploy] running next-on-pages adapter"
npx @cloudflare/next-on-pages

# Step 3 — create the Cloudflare Pages project on first deploy.
if ! npx wrangler pages project list 2>&1 | grep -q "$PROJECT_NAME"; then
    echo "[landing-deploy] creating Cloudflare Pages project '$PROJECT_NAME'"
    npx wrangler pages project create "$PROJECT_NAME" --production-branch main
fi

# Step 4 — push secrets (publishable key + API URL must be at build time;
# secret keys must be at runtime).
echo "[landing-deploy] uploading secret CLERK_SECRET_KEY"
v=$(awk -F= '/^CLERK_SECRET_KEY=/{sub("^CLERK_SECRET_KEY=",""); print; exit}' .env.local)
v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
if [ -n "$v" ]; then
    echo "$v" | npx wrangler pages secret put CLERK_SECRET_KEY --project-name "$PROJECT_NAME" >/dev/null
fi

# Step 5 — deploy.
echo "[landing-deploy] uploading build output"
npx wrangler pages deploy .vercel/output/static \
    --project-name "$PROJECT_NAME" \
    --branch main \
    --commit-dirty=true

echo "[landing-deploy] done."
echo "         Cloudflare project: https://dash.cloudflare.com/?to=/:account/pages/view/$PROJECT_NAME"
echo "         Default URL:        https://$PROJECT_NAME.pages.dev"
echo "         Custom domain:      https://app.verifiable-labs.com (after CNAME points here)"
