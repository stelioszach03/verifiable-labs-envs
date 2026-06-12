"""Result + observability schemas (SDK-safe).

These are the typed outputs every plane of the pipeline produces, so modules
built independently still connect: scores per split, contamination-adjusted
transfer metrics, a gate outcome, and a run-level observability context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _check_unit(name: str, v: float) -> None:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]; got {v}")


@dataclass(frozen=True)
class ScoreSet:
    """Scores for a candidate across the generated splits, plus risk signals.

    All scores in [0, 1]. ``dcr`` is the data-contamination-risk in [0, 1];
    ``hack_risk`` is the anti-hack violation rate in [0, 1].
    """

    public_score: float
    hidden_score: float
    ood_score: float
    adversarial_score: float
    dcr: float
    hack_risk: float
    calibration: float
    cost: float = 0.0
    latency: float = 0.0
    regression: bool = False

    def __post_init__(self) -> None:
        for n in (
            "public_score",
            "hidden_score",
            "ood_score",
            "adversarial_score",
            "dcr",
            "hack_risk",
            "calibration",
        ):
            _check_unit(n, getattr(self, n))
        if self.cost < 0 or self.latency < 0:
            raise ValueError("cost and latency must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferMetrics:
    """Contamination-adjusted transfer signals (mirrors the Lean gap track).

    These measure transfer across *generated* scenarios — explicitly NOT a
    measure of general intelligence.
    """

    public_to_hidden_gap: float
    public_to_ood_gap: float
    hidden_transfer_score: float
    ood_transfer_score: float
    adversarial_robustness_score: float
    clean_transfer_score: float
    contamination_adjusted_transfer_score: float

    @classmethod
    def from_scores(cls, s: ScoreSet) -> TransferMetrics:
        """Derive transfer metrics from a :class:`ScoreSet` deterministically."""
        p2h = s.public_score - s.hidden_score
        p2o = s.public_score - s.ood_score
        clean = s.hidden_score * (1.0 - s.dcr)
        return cls(
            public_to_hidden_gap=p2h,
            public_to_ood_gap=p2o,
            hidden_transfer_score=s.hidden_score,
            ood_transfer_score=s.ood_score,
            adversarial_robustness_score=s.adversarial_score,
            clean_transfer_score=clean,
            contamination_adjusted_transfer_score=min(clean, s.ood_score * (1.0 - s.dcr)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateOutcome:
    """A promotion-gate decision. Label is ACCEPT / REJECT / LIMITED_ROLLOUT.

    Carries the contamination-adjusted deltas so the assurance card and the
    self-improvement record can render them without recomputation.
    """

    label: str
    reasons: tuple[str, ...] = ()
    raw_vgs_delta: float | None = None
    clean_vgs_delta: float | None = None
    hidden_transfer_delta: float | None = None
    ood_transfer_delta: float | None = None
    conditions: dict[str, bool] = field(default_factory=dict)

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    LIMITED_ROLLOUT = "LIMITED_ROLLOUT"

    def __post_init__(self) -> None:
        if self.label not in (self.ACCEPT, self.REJECT, self.LIMITED_ROLLOUT):
            raise ValueError("label must be ACCEPT, REJECT, or LIMITED_ROLLOUT")

    @property
    def accepted(self) -> bool:
        return self.label == self.ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reasons": list(self.reasons),
            "raw_vgs_delta": self.raw_vgs_delta,
            "clean_vgs_delta": self.clean_vgs_delta,
            "hidden_transfer_delta": self.hidden_transfer_delta,
            "ood_transfer_delta": self.ood_transfer_delta,
            "conditions": dict(self.conditions),
        }


@dataclass(frozen=True)
class RunContext:
    """Run-level observability identifiers + telemetry (no secrets).

    Every persisted artifact references a RunContext so a gate decision, an
    assurance card, and a substrate record can be tied back to one run.
    """

    run_id: str
    mode: str
    contract_id: str | None = None
    scenario_id: str | None = None
    agent_id: str | None = None
    provider: str | None = None
    model: str | None = None
    trace_id: str | None = None
    evaluator_id: str | None = None
    score_version: str = "v0"
    gate_version: str = "v0"
    created_at: str | None = None
    cost_estimate: float | None = None
    actual_cost: float | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float | None = None
    error_status: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
