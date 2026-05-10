# V-Certified pricing (v0.0.1)

This page is the customer-facing pricing reference for the V-Certified
attestation programme. The marketing landing page is at
`/v-certified` on the Verifiable Labs site; this document is the
canonical source for sales conversations.

## Tiers

| Tier   | Cadence                            | Evidence  | Price (USD) |
|--------|------------------------------------|-----------|-------------|
| Bronze | Annual, self-attested              | 3         | $2,500/yr   |
| Silver | Annual, Vlabs-audited              | 4         | $15,000/yr  |
| Gold   | Continuous, third-party-audited    | 5+        | $45,000+/yr |

## Bronze — $2,500 / year

Self-attestation tier for early-stage AI startups, internal tools, and
non-regulated workloads. Customer uploads three artifacts (training
documentation, audit report, legal sign-off); the V-Certified review
board verifies the artifacts are well-formed and issues an X.509
certificate.

- Annual recertification
- 3 evidence artifacts (50 MB each)
- Public verification registry entry at `/v1/attestations/verify/{id}`
- Embeddable status badge SVG
- Standards crosswalks: ISO 42001 / NIST AI RMF / EU AI Act / SOC 2

**Time to issuance**: 5-10 business days.

## Silver — $15,000 / year

Vlabs-audited tier for production AI systems where customers want
third-party verification of model evaluation rigor. Adds Phase 28
monitor records (continuous capability monitoring) to the evidence
set; Verifiable Labs auditors review the monitor history before
issuance.

- Annual recertification + quarterly check-ins
- 4 evidence artifacts (Bronze + monitor records)
- Vlabs internal review board
- Continuous monitoring via Phase 28 monitors (separate billing)
- All standards crosswalks

**Time to issuance**: 15-25 business days for new customers; 5 days
on renewal.

## Gold — $45,000+ / year

Continuous tier for high-stakes AI deployments. Independent third-party
auditor (selected from our partner panel) attests on a continuous
cadence. Required for EU AI Act high-risk system conformity assessment
input.

- Continuous recertification (monthly check + annual recert)
- 5+ evidence artifacts (Silver + change-mgmt + RM/PRM records)
- Third-party audit + Vlabs internal review
- Required for EU AI Act Article 43 conformity input
- Custom crosswalk extensions on request

**Time to issuance**: 6-10 weeks for new customers; 4 weeks on
renewal.

## What's included

- One attestation per scope (per-model / per-deployment / per-org —
  D1-D scope tiers).
- Public registry entry covers the lifetime of the attestation.
- Re-issuance after revocation requires new audit cycle.
- All four standards crosswalks (no per-framework fees).
- Up to 50 MB per evidence artifact.

## What's not included

- The underlying audit work (third-party auditor fees for Gold).
- EU AI Act notified-body conformity assessment (V-Certified provides
  input artifacts; the formal assessment must be done by an EU
  notified body).
- SOC 2 attestation (V-Certified evidence can be reused as input but
  does not replace AICPA-licensed CPA-issued reports).
- ISO 42001 certification.

## Add-ons

- **Custom crosswalk extension**: $5,000 one-time. We extend a
  framework's crosswalk to cover specific clauses requested by your
  auditor.
- **Expedited issuance**: 2× tier price. Move a Bronze to 3 business
  days, Silver to 10, Gold to 4 weeks.

## Programme commitments

- 99.9% public-endpoint uptime SLA at every tier.
- Public CRL refresh every 24 h.
- Maximum 30 days from revocation request to public CRL update.

## Programme governance

V-Certified is governed by an internal review board (3 Vlabs
engineers + 1 Vlabs legal counsel) and an external advisory board
(rotating member panel; current advisors named in the company
ABOUT page once recruited). The advisory board reviews crosswalk
versions + dispute resolution + multi-party revocation requests.

For the locked v0.0.1 jurisdiction, see `docs/legal/jurisdiction.md`
(forthcoming in 31.G post-launch hardening).
