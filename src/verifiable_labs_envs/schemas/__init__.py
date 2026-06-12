"""SDK-safe shared schemas (public).

Typed data contracts that BOTH the open-source SDK/CLI and the private
platform read and emit, so modules built independently still connect:

* :mod:`.splits` — the contamination split taxonomy + policy.
* :mod:`.contract` — :class:`EvaluationContract` (the compiler that *proposes*
  one lives privately; the schema is public so the SDK can read/validate it).
* :mod:`.results` — :class:`ScoreSet`, :class:`TransferMetrics`,
  :class:`GateOutcome`, :class:`RunContext` observability.
* :mod:`.assurance_card` — :class:`AssuranceCardV2` + the exact, vetted
  formal-claim text (honesty rule).

No secrets, no hidden-eval content, no gold answers — schemas only.
"""

from __future__ import annotations

from .assurance_card import FORMAL_CLAIM, AssuranceCardV2, GateDecisionLabel
from .contract import EvaluationContract, RiskTier
from .results import GateOutcome, RunContext, ScoreSet, TransferMetrics
from .splits import Split, SplitPolicyError, validate_split_policy

__all__ = [
    "Split",
    "SplitPolicyError",
    "validate_split_policy",
    "EvaluationContract",
    "RiskTier",
    "ScoreSet",
    "TransferMetrics",
    "GateOutcome",
    "RunContext",
    "AssuranceCardV2",
    "GateDecisionLabel",
    "FORMAL_CLAIM",
]
