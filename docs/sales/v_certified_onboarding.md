# V-Certified onboarding playbook

This document is for customers who have just signed up for the
V-Certified attestation programme. It walks through the first 30 days
from sign-up to certificate issuance.

## Day 0 — sign-up

You receive a welcome email with:
- Your dashboard URL: `https://verifiable-labs.com/dashboard/attestations`
- A link to the matching tier's evidence checklist
- The internal review-board contact for your tier

## Day 1-3 — draft + scope

Decide:

1. **Scope tier** (D1-D):
   - `model` — one specific model version
   - `deployment` — a deployed product or service
   - `organization` — entire org's AI surface
2. **Tier**: Bronze / Silver / Gold (see `v_certified_pricing.md`)
3. **Standards alignment**: any subset of {ISO 42001, NIST AI RMF,
   EU AI Act, SOC 2}

Create the draft via the dashboard or API:

```bash
curl -X POST https://api.verifiable-labs.com/v1/attestations \
  -H "X-Vlabs-Key: $VLABS_KEY" \
  -d '{
    "organization": "Acme AI, Inc.",
    "scope_type": "model",
    "scope_subject": "acme-llm-v1.0",
    "tier": "silver",
    "cycle": "annual",
    "standards_requested": ["iso_42001", "nist_ai_rmf"]
  }'
```

You receive an `att_...` ID + a `vl-XXXXXXXX` public ID. The public ID
will become live on the verification registry after approval.

## Day 4-15 — evidence collection

Bronze tier evidence requirements:

- `training_doc` — Markdown / PDF documenting training data sources,
  preprocessing, lawful basis. Typical length: 5-15 pages.
- `audit_report` — Internal or external audit findings. Typical
  format: one PDF with executive summary + findings table.
- `legal_signoff` — Signed letter from your legal counsel or
  authorised signatory acknowledging fitness for purpose +
  indemnification posture.

Silver tier adds:

- `monitor_record` — JSON export from your Phase 28 monitor history.
  Use the `/v1/monitors/{id}/runs` endpoint or the dashboard download.

Gold tier adds:

- `change_mgmt` — Versioned change log spanning at least the last 12
  months of training-data + model + deployment changes.
- `rm_record` OR `prm_record` — JSON export from a Phase 29 reward
  model or Phase 30 process-reward model record. Both are accepted;
  pick whichever you have.

Upload via the dashboard UI or:

```bash
curl -X POST https://api.verifiable-labs.com/v1/attestations/{id}/artifacts \
  -H "X-Vlabs-Key: $VLABS_KEY" \
  -d '{
    "kind": "training_doc",
    "filename": "training-doc-v1.pdf",
    "content_b64": "<base64-encoded file bytes>"
  }'
```

Files are SHA-256 hashed at upload + capped at 50 MB decoded. Sensitive
files can be marked `encrypted: true` (you encrypt client-side; we
store the ciphertext + the flag for the auditor).

## Day 16-25 — submission + review

Submit the attestation:

```bash
curl -X PATCH https://api.verifiable-labs.com/v1/attestations/{id} \
  -H "X-Vlabs-Key: $VLABS_KEY" \
  -d '{"action": "submit"}'
```

The status transitions to `submitted`. The review-board reviews per
your tier:

- Bronze: Vlabs internal review (5 business days).
- Silver: Vlabs internal review + monitor-record sanity check (10
  business days).
- Gold: external auditor (4-6 weeks) + Vlabs internal review (1 week).

Possible outcomes:

- `approve` — issuance + X.509 cert + public registry entry.
- `reject` — attestation moves to `withdrawn`. You can start a new
  draft.
- `request_more` — attestation moves back to `draft`. You upload
  additional evidence + resubmit. Typically resolved within 5
  business days.

## Day 26+ — issuance + verification

On approval the dashboard shows:

- `cert_serial` — opaque V-Certified serial (32-char hex with
  `stub-` prefix in v0.0.1).
- `certificate_pem` — X.509 leaf certificate PEM.
- `expires_at` — issuance + 365 days (annual) or 395 days
  (continuous).

Public verifiers can hit:

```bash
# Verify by public_id (most common)
curl https://api.verifiable-labs.com/v1/attestations/verify/vl-XXXXXXXX

# Verify by cert serial (when only the cert is in hand)
curl https://api.verifiable-labs.com/v1/attestations/verify-by-cert/{cert_serial}

# Embed a status badge
<img src="https://api.verifiable-labs.com/v1/attestations/badge/vl-XXXXXXXX.svg" />
```

## Day 60+ — renewal preparation

Annual tiers (Bronze + Silver) get a 30-day pre-expiry email reminder.
Gold's continuous tier auto-renews monthly with no action needed
unless a monitor flag triggers a request-more.

To pre-emptively initiate a renewal:

```bash
curl -X POST https://api.verifiable-labs.com/v1/attestations/{id}/renew \
  -H "X-Vlabs-Key: $VLABS_KEY" \
  -d '{"idempotency_key": "renew-2026-q2"}'
```

The idempotency key protects against double-submits within a 24 h
window.

## Communications cadence

- **Sign-up confirmation** — same day.
- **Reminder if no draft created in 7 days** — 1 email.
- **Reminder if no submission in 30 days from draft creation** — 1 email.
- **Decision notification** — same day as audit decision.
- **Renewal reminder** — 60 + 30 + 7 days before expiry.

## Escalations

If any step blocks for more than its expected duration, email
`v-certified@verifiable-labs.com` with the public_id. We aim to
respond within one business day.
