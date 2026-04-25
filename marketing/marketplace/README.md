# marketplace landing page (vision)

Single-file static HTML page describing the Verifiable Labs
marketplace **vision** for v0.3. **This is not a product page.** The
marketplace itself doesn't exist yet — the page is a public artifact
of where the project intends to go.

## What's on it

- Hero with the v0.3 framing ("planned for v0.3")
- "Why a marketplace" — the procedural-regeneration / conformal-
  calibration pitch
- Three roles: contributors / consumers / enterprise
- 2026 roadmap with explicit `SHIPPED / PLANNED / SPECULATIVE` tags
- Email capture form that opens a `mailto:` to the founder (no
  backend, no SaaS list, no third parties)

## Why static HTML + CDN Tailwind

The page has no JS framework, no build step, and no backend. One
`index.html`, two stylesheet sources (Tailwind CDN + Google Fonts),
and a `mailto:` form. The whole thing fits in one file.

This is **deliberate**. The marketplace is a vision document; we
don't want to own ongoing maintenance of a marketing site while the
real platform is the focus.

When the marketplace is real (v0.3 at earliest), this page will be
replaced with a proper deployment.

## Local preview

```bash
python3 -m http.server 8765 --directory marketing/marketplace
open http://localhost:8765
```

## Deploy options

The page is one HTML file. Drop it on:

- **Vercel** — `vercel deploy` from this directory
- **Netlify Drop** — drag-and-drop the directory
- **GitHub Pages** — set the source to `marketing/marketplace/` (or
  copy the file into a separate repo)
- **Cloudflare Pages** — connect the repo, set build output to this
  directory

## Tailwind CDN warning

The console shows a Tailwind warning that the CDN should not be used
in production. For an alpha-stage vision page this is acceptable —
the brief explicitly specified Tailwind via CDN, and the page is
not a high-traffic destination. v0.3, when the marketplace is real,
will use a proper Tailwind build.

## What this page does NOT claim

- It does **not** assert the marketplace exists. The phrase "planned
  for v0.3" appears prominently.
- It does **not** collect emails into a database. The form is
  `mailto:`, full stop.
- It does **not** advertise pricing, contracts, or paid features. v0.1
  is alpha and free; everything else is roadmap.
- It does **not** make regulatory-compliance claims on the platform's
  behalf.

These choices are intentional and match the project's stated
honesty discipline.
