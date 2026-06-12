# V-Certified ↔ SOC 2 Trust Services Criteria crosswalk

**Crosswalk version**: `v0.0.1` (Phase 31.E)
**Framework version**: AICPA TSC 2017 (with 2022 revisions)

SOC 2 reports issued by AICPA-licensed CPAs cover service
organisations' controls relevant to the Trust Services Categories:
Security, Availability, Processing Integrity, Confidentiality, and
Privacy. The Common Criteria (CC1-CC9) form the baseline; AI-specific
add-on points of focus extend several criteria for AI workloads.

V-Certified attestations contribute documentary evidence supporting
several CC criteria, particularly around data provenance + control
monitoring + change management.

## How to use this crosswalk

The full machine-readable mapping is served at:

    GET /v1/standards/soc2

V-Certified does NOT issue a SOC 2 report. Customers seeking a SOC 2
attestation must engage an AICPA-licensed CPA firm. V-Certified
evidence may be reused as input to the SOC 2 audit.

## Control mapping

| V-Certified control | Title | SOC 2 CC criteria | Evidence kinds |
|---------------------|-------|-------------------|----------------|
| VC-1.1 | AI training-data provenance | CC1.1, CC2.1, CC6.1 | training_doc, audit_report |
| VC-2.1 | Model evaluation rigor | CC4.1, CC4.2 | monitor_record, rm_record, prm_record |
| VC-3.1 | Change management trail | CC8.1 | change_mgmt |
| VC-4.1 | Legal accountability | CC1.2 | legal_signoff |

## CC-by-CC

- **CC1 (control environment)**: VC-1.1 + VC-4.1 cover CC1.1 (commitment
  to integrity), CC1.2 (board oversight), and CC2.1 (information
  quality).
- **CC4 (monitoring activities)**: VC-2.1 directly satisfies CC4.1
  (continuous monitoring) + CC4.2 (deviation evaluation) through Phase
  28 monitor + reward-model records.
- **CC6 (logical access)**: VC-1.1's data-provenance trace contributes
  to CC6.1 (logical access restrictions) by establishing chain of
  custody for training data.
- **CC8 (change management)**: VC-3.1 directly satisfies CC8.1 with
  the Tier-Gold change-management artifact.

## SOC 2 Type 1 vs Type 2

V-Certified Bronze (annual self-attested) approximates a SOC 2 Type 1
posture (point-in-time design effectiveness). V-Certified Silver +
Gold (continuous monitoring) approximate Type 2 (operating
effectiveness over a period) by virtue of Phase 28 monitor records
covering the audit window.

## Conformance level

V-Certified attestation does NOT confer SOC 2. Customers seeking a
SOC 2 report must follow AICPA SSAE 18 attestation standards with a
licensed CPA firm.
