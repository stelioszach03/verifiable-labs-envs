"""Assurance Card v2 schema + the vetted formal-claim text (SDK-safe).

The card is the human- and machine-readable verdict of a run. ``FORMAL_CLAIM``
is the ONLY approved wording about formal verification; every card, doc, and
export reuses this exact string so the honesty rule cannot drift. It must
never be reworded into "formally verified API/system/product/code".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# The single source of truth for the formal-verification claim (honesty rule).
FORMAL_CLAIM = (
    "Selected mathematical properties behind the contamination-resistant "
    "promotion gate are machine-verified in Lean 4. The implementation is "
    "property-tested against the formal specification."
)

# Phrases that must never appear anywhere (asserted by the forbidden-claims
# grep in T10). Exposed here so tests can import the canonical list.
FORBIDDEN_CLAIMS = (
    "formally verified system",  # never allowed
    "formally verified api",  # never allowed
    "formally verified product",  # never allowed
    "formally verified code",  # never allowed
    "formally verified service",  # never allowed
    "we prove the model generalizes",
    "we eliminate contamination",
    "guarantee general intelligence",
    "prove agi safety",
)


class GateDecisionLabel(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    LIMITED_ROLLOUT = "LIMITED_ROLLOUT"


@dataclass(frozen=True)
class AssuranceCardV2:
    """Run verdict. Carries scores, transfer, gate decision, and the formal
    scope — plus redaction/export-safety flags so it is never published with
    private content by accident."""

    card_version: str
    run_id: str
    agent_id: str | None
    baseline_id: str | None
    candidate_id: str | None
    decision: str
    raw_vgs: float | None = None
    dcr: float | None = None
    clean_vgs: float | None = None
    public_score: float | None = None
    hidden_score: float | None = None
    ood_score: float | None = None
    generalization_gap: float | None = None
    hack_risk: float | None = None
    calibration: float | None = None
    cost: float | None = None
    latency: float | None = None
    regression: bool = False
    reject_reasons: tuple[str, ...] = ()
    formal_claim: str = FORMAL_CLAIM
    formal_scope: str = FORMAL_CLAIM
    contamination_policy: str = "generated_after_freeze"
    redaction_status: str = "unredacted"
    hf_public_safe: bool = False
    wandb_run_url: str | None = None
    hf_dataset_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {d.value for d in GateDecisionLabel}:
            raise ValueError("decision must be ACCEPT, REJECT, or LIMITED_ROLLOUT")
        # Honesty guard: the formal claim must be the vetted text verbatim.
        if self.formal_claim != FORMAL_CLAIM:
            raise ValueError("formal_claim must be the vetted FORMAL_CLAIM text")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reject_reasons"] = list(self.reject_reasons)
        return d
