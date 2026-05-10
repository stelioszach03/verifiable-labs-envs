"""V-Certified standards crosswalks (Phase 31.E).

Maps the V-Certified control set onto four external compliance
frameworks. Each crosswalk is a static, version-pinned JSON-shaped
mapping of ``vc_control_id -> {framework_clauses, evidence_kinds,
notes}``. The version is recorded on every attestation at issuance
(R1 mitigation: ``standards_alignment.crosswalk_version`` is frozen
when an audit-decision approval lands so verifiers always see the
crosswalk that was current at the time of issuance, not whatever
crosswalk version is current today).

Four supported frameworks (D8 locked subset):
- ``iso_42001`` — ISO/IEC 42001 AI management system clauses A.5-A.10.
- ``nist_ai_rmf`` — NIST AI RMF Core Functions (Govern / Map /
  Measure / Manage).
- ``eu_ai_act`` — EU AI Act high-risk system controls (Annex IV +
  Article 9 risk management + Article 10 data governance).
- ``soc2`` — SOC 2 Trust Services Criteria CC1-CC9 + AI add-on.
"""
from __future__ import annotations

from vlabs_api.standards.crosswalks import (
    CROSSWALK_VERSION,
    KNOWN_FRAMEWORKS,
    Crosswalk,
    CrosswalkEntry,
    get_crosswalk,
    list_all_crosswalks,
)

__all__ = [
    "CROSSWALK_VERSION",
    "KNOWN_FRAMEWORKS",
    "Crosswalk",
    "CrosswalkEntry",
    "get_crosswalk",
    "list_all_crosswalks",
]
