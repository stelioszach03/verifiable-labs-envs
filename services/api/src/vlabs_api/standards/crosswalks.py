"""V-Certified ↔ external-framework crosswalks (Phase 31.E).

Static mapping data, no I/O — the crosswalks are baked into the
deployment so verifiers don't need a network round-trip past the
public API.

Schema:
    Crosswalk = list[CrosswalkEntry]
    CrosswalkEntry = {
        vc_control_id: str,           # V-Certified control number
        vc_control_title: str,
        framework_clauses: list[str], # external clauses this maps to
        evidence_kinds: list[str],    # the D9 artifact kinds providing evidence
        notes: str | None,
    }

Coverage per framework (v0.0.1) — intentionally narrow; broader
clause coverage is a 31.E.2 follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass

CROSSWALK_VERSION: str = "v0.0.1"
"""Bumped together for all four crosswalks; recorded at attestation
issuance for R1 mitigation."""

KNOWN_FRAMEWORKS: tuple[str, ...] = (
    "iso_42001",
    "nist_ai_rmf",
    "eu_ai_act",
    "soc2",
)
"""Locked subset matching :data:`vlabs_api.attestation_service.ALLOWED_
STANDARDS`. Extending this requires bumping :data:`CROSSWALK_VERSION`."""


@dataclass(frozen=True)
class CrosswalkEntry:
    vc_control_id: str
    vc_control_title: str
    framework_clauses: tuple[str, ...]
    evidence_kinds: tuple[str, ...]
    notes: str | None = None


Crosswalk = tuple[CrosswalkEntry, ...]


# ── ISO/IEC 42001 ──────────────────────────────────────────────────


_ISO_42001: Crosswalk = (
    CrosswalkEntry(
        vc_control_id="VC-1.1",
        vc_control_title="AI training-data provenance",
        framework_clauses=("A.5.2", "A.6.1", "A.7.1"),
        evidence_kinds=("training_doc", "audit_report"),
        notes="Provenance documentation + 3rd-party audit attests "
        "data origin + lawful-basis records.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-2.1",
        vc_control_title="Model evaluation rigor",
        framework_clauses=("A.7.4", "A.8.2"),
        evidence_kinds=("monitor_record", "rm_record", "prm_record"),
        notes="Continuous monitor records + reward-model records "
        "satisfy ISO 42001 evaluation requirements.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-3.1",
        vc_control_title="Change management trail",
        framework_clauses=("A.9.1", "A.9.2"),
        evidence_kinds=("change_mgmt",),
        notes="Tier-Gold required artifact aligns with ISO 42001 "
        "change-control requirements.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-4.1",
        vc_control_title="Legal accountability",
        framework_clauses=("A.10.1",),
        evidence_kinds=("legal_signoff",),
        notes="Customer legal sign-off attests to fitness for "
        "intended use + indemnification posture.",
    ),
)


# ── NIST AI RMF ────────────────────────────────────────────────────


_NIST_AI_RMF: Crosswalk = (
    CrosswalkEntry(
        vc_control_id="VC-1.1",
        vc_control_title="AI training-data provenance",
        framework_clauses=("MAP-2.3", "MAP-3.4"),
        evidence_kinds=("training_doc", "audit_report"),
        notes="MAP function: data provenance + characterisation.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-2.1",
        vc_control_title="Model evaluation rigor",
        framework_clauses=("MEASURE-1.1", "MEASURE-2.5", "MEASURE-3.2"),
        evidence_kinds=("monitor_record", "rm_record", "prm_record"),
        notes="MEASURE function: continuous evaluation + uncertainty "
        "quantification.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-3.1",
        vc_control_title="Change management trail",
        framework_clauses=("MANAGE-3.1", "MANAGE-3.2"),
        evidence_kinds=("change_mgmt",),
        notes="MANAGE function: post-deployment risk monitoring + "
        "change control.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-5.1",
        vc_control_title="Programme governance",
        framework_clauses=("GOVERN-1.1", "GOVERN-2.1", "GOVERN-3.2"),
        evidence_kinds=("legal_signoff", "third_party_audit"),
        notes="GOVERN function: organisational accountability + "
        "external audit posture.",
    ),
)


# ── EU AI Act ──────────────────────────────────────────────────────


_EU_AI_ACT: Crosswalk = (
    CrosswalkEntry(
        vc_control_id="VC-1.1",
        vc_control_title="AI training-data provenance",
        framework_clauses=("Article 10",),
        evidence_kinds=("training_doc",),
        notes="Article 10 high-risk system data governance.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-2.1",
        vc_control_title="Model evaluation rigor",
        framework_clauses=("Article 9", "Article 15"),
        evidence_kinds=("monitor_record", "rm_record", "prm_record"),
        notes="Article 9 risk management + Article 15 robustness.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-3.1",
        vc_control_title="Change management trail",
        framework_clauses=("Article 17", "Annex IV §2(g)"),
        evidence_kinds=("change_mgmt",),
        notes="Article 17 quality management + Annex IV technical "
        "documentation requirements.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-6.1",
        vc_control_title="Conformity assessment",
        framework_clauses=("Article 43", "Annex VII"),
        evidence_kinds=("third_party_audit",),
        notes="Article 43 conformity assessment by notified body.",
    ),
)


# ── SOC 2 ──────────────────────────────────────────────────────────


_SOC2: Crosswalk = (
    CrosswalkEntry(
        vc_control_id="VC-1.1",
        vc_control_title="AI training-data provenance",
        framework_clauses=("CC1.1", "CC2.1", "CC6.1"),
        evidence_kinds=("training_doc", "audit_report"),
        notes="CC1 control environment + CC6 logical access "
        "(data provenance traces).",
    ),
    CrosswalkEntry(
        vc_control_id="VC-2.1",
        vc_control_title="Model evaluation rigor",
        framework_clauses=("CC4.1", "CC4.2"),
        evidence_kinds=("monitor_record", "rm_record", "prm_record"),
        notes="CC4 monitoring of controls — Phase 28 monitor records "
        "directly satisfy continuous-monitoring evidence.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-3.1",
        vc_control_title="Change management trail",
        framework_clauses=("CC8.1",),
        evidence_kinds=("change_mgmt",),
        notes="CC8 change management.",
    ),
    CrosswalkEntry(
        vc_control_id="VC-4.1",
        vc_control_title="Legal accountability",
        framework_clauses=("CC1.2",),
        evidence_kinds=("legal_signoff",),
        notes="CC1.2 board-level accountability + sign-off.",
    ),
)


_CROSSWALKS: dict[str, Crosswalk] = {
    "iso_42001": _ISO_42001,
    "nist_ai_rmf": _NIST_AI_RMF,
    "eu_ai_act": _EU_AI_ACT,
    "soc2": _SOC2,
}


# ── public API ─────────────────────────────────────────────────────


def get_crosswalk(framework: str) -> Crosswalk:
    """Return the locked-version crosswalk for ``framework``.

    Raises :class:`KeyError` for unknown frameworks (callers must
    pre-validate against :data:`KNOWN_FRAMEWORKS`).
    """
    return _CROSSWALKS[framework]


def list_all_crosswalks() -> dict[str, Crosswalk]:
    """Return a shallow copy mapping framework -> crosswalk."""
    return dict(_CROSSWALKS)
