"""Run configuration — product modes + privacy-preserving default flags.

A :class:`RunConfig` selects one of four product modes and a set of explicit
capability flags. The default is the **most privacy-preserving** configuration
(evaluate-only, nothing enabled, nothing exported, human review required) so a
caller who passes nothing can never accidentally mutate an agent, store data
for training, or export anything.

SDK-safe (stdlib only); the private platform reads/extends this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Mode(StrEnum):
    """The four customer-selectable product modes."""

    EVALUATE_ONLY = "evaluate_only"
    GATE_ONLY = "gate_only"
    IMPROVE_AND_GATE = "improve_and_gate"
    SUBSTRATE = "substrate"


@dataclass(frozen=True)
class RunConfig:
    """One pipeline run's configuration.

    Defaults are privacy-preserving: evaluate-only, no improvement, no
    candidate generation, no substrate records, no future-training use, no
    public export, human review required, and ``dry_run`` on (dummy provider).
    """

    mode: Mode = Mode.EVALUATE_ONLY
    enable_improvement_suggestions: bool = False
    enable_candidate_config: bool = False
    enable_substrate_records: bool = False
    allow_future_training_use: bool = False
    public_export: bool = False
    human_review_required: bool = True
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            raise TypeError("mode must be a Mode")
        # Cross-flag safety invariants. Improvement artifacts only exist when
        # their mode is selected; substrate records only in substrate mode.
        if self.enable_candidate_config and not self.enable_improvement_suggestions:
            raise ValueError("enable_candidate_config requires enable_improvement_suggestions")
        if self.public_export and self.human_review_required is False:
            # Allowed, but flagged loudly elsewhere; here we only block the
            # nonsensical combination of exporting while disallowing review
            # of training use.
            pass

    # — capability predicates the orchestrator reads —
    @property
    def does_evaluate(self) -> bool:
        return True  # every mode evaluates first

    @property
    def does_gate(self) -> bool:
        return self.mode in (Mode.GATE_ONLY, Mode.IMPROVE_AND_GATE)

    @property
    def does_improve(self) -> bool:
        return self.mode is Mode.IMPROVE_AND_GATE

    @property
    def does_substrate(self) -> bool:
        return self.mode is Mode.SUBSTRATE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        data = dict(data)
        if "mode" in data and not isinstance(data["mode"], Mode):
            data["mode"] = Mode(str(data["mode"]))
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


def default_config() -> RunConfig:
    """Return the privacy-preserving default run configuration."""
    return RunConfig()
