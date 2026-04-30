# vlabs-landing — marketing site + dashboard

Next.js 15 + TypeScript + Tailwind + Clerk auth. Phase 16 Stage B
deliverable: signup/login, API key generation, usage view, Stripe
upgrade flow (test mode only).

> **Status: 0.0.1 alpha.** Stage B ships the code. Cloudflare Pages
> deploy + custom domain land in Stage C.

## Routes

| Path | Purpose |
|---|---|
| `/` | marketing landing (hero + features + CTA) |
| `/pricing` | three-tier pricing card grid |
| `/sign-in` | Clerk-hosted sign-in |
| `/sign-up` | Clerk-hosted sign-up |
| `/dashboard` | overview after auth |
| `/dashboard/api-keys` | mint / list / revoke API keys |
| `/dashboard/usage` | current month usage + tier quota |
| `/dashboard/billing` | upgrade button + Stripe portal link |

## Local development

```bash
cd services/landing
cp .env.example .env.local
# fill in NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY + CLERK_SECRET_KEY
# (and run vlabs-api on :8000 in another terminal)

npm install
npm run dev          # http://localhost:3000
```

The dashboard talks to `vlabs-api` via `NEXT_PUBLIC_VLABS_API_URL`
(defaults to `http://localhost:8000`). Server-side mutations
(`/v1/billing/checkout`, `/v1/keys`) are forwarded with the Clerk
session token via Next.js server actions.

## Deploy

Cloudflare Pages (Stage C):

```bash
npm run build
npx @cloudflare/next-on-pages
# wrangler pages deploy .vercel/output/static
```

## Structure

```
services/landing/
├── app/
│   ├── layout.tsx                # ClerkProvider + Tailwind globals
│   ├── globals.css
│   ├── page.tsx                  # marketing landing
│   ├── pricing/page.tsx
│   ├── sign-in/[[...sign-in]]/page.tsx
│   ├── sign-up/[[...sign-up]]/page.tsx
│   └── dashboard/
│       ├── layout.tsx            # signed-in guard
│       ├── page.tsx              # overview
│       ├── api-keys/page.tsx
│       ├── usage/page.tsx
│       ├── billing/page.tsx
│       └── actions.ts            # server actions (POST /v1/keys, /v1/billing/*)
├── lib/api.ts                    # typed fetch wrapper for vlabs-api
├── middleware.ts                 # Clerk middleware
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.mjs
└── tsconfig.json
```
