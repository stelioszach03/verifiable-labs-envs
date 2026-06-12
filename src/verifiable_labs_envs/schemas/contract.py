"""EvaluationContract schema (SDK-safe).

A contract captures what a customer wants verified about an agent. The
private platform's compiler *proposes* a contract (LLM may draft, validation
decides); this public dataclass is the typed artifact both sides exchange so
the SDK/CLI can read, validate, and emit contracts deterministically.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .splits import Split


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def _hash_sources(sources: list[str]) -> tuple[str, ...]:
    """Hash each source string (we store hashes, never raw customer text)."""
    return tuple(hashlib.sha256(s.encode()).hexdigest() for s in sources)


@dataclass(frozen=True)
class EvaluationContract:
    """Typed evaluation contract.

    ``source_hashes`` holds SHA-256 hashes of the inputs the contract was
    compiled from (goal/docs/traces) — never the raw text — so a contract is
    auditable without leaking customer data.
    """

    contract_id: str
    capability: str
    domain: str
    user_goal: str
    success_criteria: tuple[str, ...] = ()
    safety_constraints: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    required_splits: tuple[Split, ...] = (Split.PUBLIC_DEMO, Split.HIDDEN_EVAL)
    required_verifiers: tuple[str, ...] = ()
    risk_tier: RiskTier = RiskTier.MEDIUM
    contamination_policy: str = "generated_after_freeze"
    generated_after_freeze_required: bool = True
    public_release_allowed: bool = False
    train_allowed: bool = False
    hidden_eval_allowed: bool = True
    created_at: str | None = None
    source_hashes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id is required")
        if not self.capability:
            raise ValueError("capability is required")
        if not self.user_goal:
            raise ValueError("user_goal is required")
        if not isinstance(self.risk_tier, RiskTier):
            raise TypeError("risk_tier must be a RiskTier")
        for s in self.required_splits:
            if not isinstance(s, Split):
                raise TypeError("required_splits must contain Split values")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_tier"] = self.risk_tier.value
        d["required_splits"] = [s.value for s in self.required_splits]
        return d

    @classmethod
    def from_sources(
        cls,
        *,
        contract_id: str,
        capability: str,
        domain: str,
        user_goal: str,
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationContract:
        """Build a contract, hashing raw inputs into ``source_hashes``."""
        return cls(
            contract_id=contract_id,
            capability=capability,
            domain=domain,
            user_goal=user_goal,
            source_hashes=_hash_sources(sources or []),
            **kwargs,
        )
