# V-Certified ↔ EU AI Act crosswalk

**Crosswalk version**: `v0.0.1` (Phase 31.E)
**Framework version**: Regulation (EU) 2024/1689 — published OJ L,
adopted 13 June 2024, with staggered application dates 2025-2027.

The EU AI Act is the world's first horizontal regulation of AI
systems, establishing risk-based requirements with the strictest rules
applying to high-risk systems (Annex III) and general-purpose AI
models with systemic risk.

V-Certified is **not** a notified-body conformity assessment. For
high-risk AI systems requiring third-party conformity assessment under
Article 43, customers must engage an EU-recognised notified body.
V-Certified evidence can be used as input to that assessment but does
not replace it.

## How to use this crosswalk

The full machine-readable mapping is served at:

    GET /v1/standards/eu_ai_act

## Control mapping

| V-Certified control | Title | EU AI Act articles | Evidence kinds |
|---------------------|-------|---------------------|----------------|
| VC-1.1 | AI training-data provenance | Article 10 | training_doc |
| VC-2.1 | Model evaluation rigor | Article 9, Article 15 | monitor_record, rm_record, prm_record |
| VC-3.1 | Change management trail | Article 17, Annex IV §2(g) | change_mgmt |
| VC-6.1 | Conformity assessment | Article 43, Annex VII | third_party_audit |

## Article-by-article

- **Article 9 (risk management system)**: V-Certified Silver + Gold
  attestations require continuous risk-tracking via Phase 28 monitors,
  satisfying the iterative + lifecycle nature of Article 9.
- **Article 10 (data and data governance)**: VC-1.1's training-data
  provenance documentation directly addresses Article 10's
  representativeness, completeness + relevance + bias-checking
  requirements.
- **Article 15 (accuracy, robustness, cybersecurity)**: VC-2.1's
  reward/process-reward evidence documents accuracy + uncertainty
  quantification.
- **Article 17 (quality management system)**: VC-3.1's change
  management records satisfy the QMS documentation requirement.
- **Article 43 + Annex VII (conformity assessment)**: VC-6.1's
  third-party audit artifact documents the underlying conformity
  assessment performed by a notified body.

## Conformance level

V-Certified attestation does NOT confer EU AI Act conformity. Customers
deploying high-risk AI systems in the EU must complete the conformity
assessment procedures specified in Articles 43-49 of the Act with an
EU-notified body. V-Certified evidence is supplementary documentation
that auditors and notified bodies may accept as input.

## Geographic note

V-Certified v0.0.1 is a US-Delaware-jurisdiction programme (per
PHASE_31_PLAN.md §4 D14-A). The EU AI Act crosswalk is informational
guidance for customers operating EU-deployed AI systems; we do not
yet operate as an EU-recognised certification body.
