# V-Certified ↔ NIST AI Risk Management Framework crosswalk

**Crosswalk version**: `v0.0.1` (Phase 31.E)
**Framework version**: NIST AI RMF 1.0 (January 2023)

The NIST AI Risk Management Framework is a voluntary US-government
guidance document organising AI risk management around four Core
Functions: Govern, Map, Measure, Manage.

V-Certified attestations directly satisfy several Map + Measure +
Manage activities through the evidence artifacts they require.

## How to use this crosswalk

The full machine-readable mapping is served at:

    GET /v1/standards/nist_ai_rmf

Customers using the V-Certified programme as part of NIST AI RMF
adoption should treat this crosswalk as a guide; final compliance
determinations remain with the deploying organisation.

## Control mapping

| V-Certified control | Title | NIST AI RMF subcategories | Evidence kinds |
|---------------------|-------|---------------------------|----------------|
| VC-1.1 | AI training-data provenance | MAP-2.3, MAP-3.4 | training_doc, audit_report |
| VC-2.1 | Model evaluation rigor | MEASURE-1.1, MEASURE-2.5, MEASURE-3.2 | monitor_record, rm_record, prm_record |
| VC-3.1 | Change management trail | MANAGE-3.1, MANAGE-3.2 | change_mgmt |
| VC-5.1 | Programme governance | GOVERN-1.1, GOVERN-2.1, GOVERN-3.2 | legal_signoff, third_party_audit |

## Function-by-function

- **GOVERN** (organisational accountability): VC-5.1 covers GOVERN-1.1
  (legal + regulatory requirements understood), GOVERN-2.1 (roles +
  responsibilities clear), and GOVERN-3.2 (workforce capability +
  diversity).
- **MAP** (context characterisation): VC-1.1 covers MAP-2.3 (data set
  characterisation) and MAP-3.4 (intended + unintended end users
  mapped).
- **MEASURE** (continuous evaluation): VC-2.1 covers MEASURE-1.1
  (metrics established), MEASURE-2.5 (quantitative analyses
  documented), and MEASURE-3.2 (risks tracked over time).
- **MANAGE** (post-deployment): VC-3.1 covers MANAGE-3.1
  (deployment-relevant risks tracked) and MANAGE-3.2 (change
  management).

## Conformance level

V-Certified attestation provides documentary evidence supporting NIST
AI RMF conformance. The framework itself is voluntary and self-attested
— there is no formal certification body, so V-Certified can serve as a
third-party verification step.
