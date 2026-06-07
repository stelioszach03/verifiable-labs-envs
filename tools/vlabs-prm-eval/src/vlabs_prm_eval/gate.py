"""Eval-card → ``ModelMetrics`` mapping + gate evaluator.

The 7-condition ``AcceptUpdate`` predicate itself lives in
``verifiable_labs_envs.formal_spec.gate`` (machine-verified in Lean).
This module only handles the schema bridge from a process-reward-model
eval card (JSON dict) to ``ModelMetrics`` and the user-facing rendering.

Eval-card schema (canonical fields, all optional unless noted):

    {
      "model_id":                "<str, required>",
      "processbench_overall":    <float in [0,1]>,
      "mean_held_out_env_metric":<float in [0,1]>,
      "bon_accuracy":            <float in [0,1]>,
      "coverage":                <float in [0,1]>,
      "held_out_envs":           { "<env_name>": <float>, ... },
      "invariance_violation_rate": <float in [0,1]>,
      "cost":                    <float, $ per audit>,
      "latency":                 <float, seconds per audit>
    }

The mapping is documented in ``DEFAULT_METRICS_MAP`` and overridable
via the CLI's ``--metrics-map <path>`` flag.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiable_labs_envs.formal_spec.gate import (
    GateDecision,
    ModelMetrics,
    Tolerances,
    accept_update,
)

# ---------------------------------------------------------------------
# Default eval-card → ModelMetrics weights
# ---------------------------------------------------------------------
DEFAULT_METRICS_MAP: dict[str, Any] = {
    "vgs_weights": {
        # Sum = 1.0. Override per checkpoint-promotion policy via
        # --metrics-map.
        "processbench_overall": 0.40,
        "mean_held_out_env_metric": 0.30,
        "bon_accuracy": 0.30,
    },
    "regression_rules": {
        # Trigger ``regression=True`` if any of these fire.
        "processbench_min": 0.60,
        "bon_lift_min": 0.05,
        "coverage_target": 0.90,
        "coverage_tol": 0.05,
    },
}


@dataclass(frozen=True)
class EvalCard:
    """Minimal eval-card view used by the gate. Extra fields are tolerated."""

    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> EvalCard:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
        return cls(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


@dataclass(frozen=True)
class MappingReport:
    """Diagnostic for the eval-card → ModelMetrics bridge."""

    warnings: tuple[str, ...] = field(default_factory=tuple)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected number, got {value!r}")
    return float(value)


def card_to_metrics(
    card: EvalCard,
    metrics_map: Mapping[str, Any] | None = None,
    *,
    cost_override: float | None = None,
    latency_override: float | None = None,
) -> tuple[ModelMetrics, MappingReport]:
    """Map an eval card onto ``ModelMetrics``.

    Returns the metrics plus a report of any warnings (e.g. missing
    invariance estimator). Raises ``ValueError`` on hard schema errors
    (e.g. negative cost, missing required fields).
    """
    m = dict(DEFAULT_METRICS_MAP)
    if metrics_map is not None:
        m = {**m, **metrics_map}
    weights = m["vgs_weights"]
    regression_rules = m["regression_rules"]

    warnings: list[str] = []

    # ---- VGS composite ----
    components = {
        "processbench_overall": _to_float(
            "processbench_overall", card.get("processbench_overall", 0.0)
        ),
        "mean_held_out_env_metric": _to_float(
            "mean_held_out_env_metric", card.get("mean_held_out_env_metric", 0.0)
        ),
        "bon_accuracy": _to_float("bon_accuracy", card.get("bon_accuracy", 0.0)),
    }
    if card.get("mean_held_out_env_metric") is None:
        held_out = card.get("held_out_envs") or {}
        if isinstance(held_out, dict) and held_out:
            components["mean_held_out_env_metric"] = _mean(
                [_to_float(k, v) for k, v in held_out.items()]
            )

    weight_sum = sum(weights.values())
    if not (0.99 <= weight_sum <= 1.01):
        warnings.append(
            f"vgs_weights sum to {weight_sum:.4f}, expected 1.0; "
            "vgs will be unnormalised."
        )
    vgs_value = sum(components[name] * weight for name, weight in weights.items())

    # ---- hack_risk from invariance violation rate ----
    inv = card.get("invariance_violation_rate")
    if inv is None:
        warnings.append(
            "card has no `invariance_violation_rate`; using hack_risk=0.0. "
            "Run vlabs-prm-eval's invariance harness to populate it; the "
            "gate cannot detect shortcuts without it."
        )
        hack_risk = 0.0
    else:
        hack_risk = _to_float("invariance_violation_rate", inv)

    # ---- calibration ← coverage ----
    calibration = _to_float("coverage", card.get("coverage", 0.0))

    # ---- ood ← mean of held_out_envs ----
    held_out = card.get("held_out_envs") or {}
    if isinstance(held_out, dict) and held_out:
        ood = _mean([_to_float(k, v) for k, v in held_out.items()])
    else:
        ood = components["mean_held_out_env_metric"]

    # ---- cost / latency ----
    cost = (
        cost_override
        if cost_override is not None
        else _to_float("cost", card.get("cost", 0.0))
    )
    latency = (
        latency_override
        if latency_override is not None
        else _to_float("latency", card.get("latency", 0.0))
    )

    # ---- regression flag ----
    pb = components["processbench_overall"]
    bon = components["bon_accuracy"]
    coverage_target = regression_rules["coverage_target"]
    coverage_tol = regression_rules["coverage_tol"]
    regression = bool(
        pb < regression_rules["processbench_min"]
        or abs(calibration - coverage_target) > coverage_tol
        or bon < 0.0  # placeholder — actual BoN-lift rule needs the OLD card too
    )

    return (
        ModelMetrics(
            vgs=vgs_value,
            hack_risk=hack_risk,
            calibration=calibration,
            ood=ood,
            cost=cost,
            latency=latency,
            regression=regression,
        ),
        MappingReport(warnings=tuple(warnings)),
    )


def evaluate_gate(
    old_card_path: str | Path,
    new_card_path: str | Path,
    tol: Tolerances,
    metrics_map_path: str | Path | None = None,
    *,
    cost_old: float | None = None,
    cost_new: float | None = None,
    latency_old: float | None = None,
    latency_new: float | None = None,
) -> tuple[GateDecision, ModelMetrics, ModelMetrics, list[str]]:
    """Load two cards, map to metrics, evaluate gate. Return (decision,
    old_metrics, new_metrics, warnings)."""

    metrics_map: Mapping[str, Any] | None = None
    if metrics_map_path is not None:
        with open(metrics_map_path, encoding="utf-8") as f:
            metrics_map = json.load(f)

    old_card = EvalCard.load(old_card_path)
    new_card = EvalCard.load(new_card_path)

    old, report_old = card_to_metrics(
        old_card, metrics_map, cost_override=cost_old, latency_override=latency_old
    )
    new, report_new = card_to_metrics(
        new_card, metrics_map, cost_override=cost_new, latency_override=latency_new
    )

    warnings = [*report_old.warnings, *report_new.warnings]

    # Re-derive regression using the OLD bon_accuracy too (the rule
    # requires lift vs the previous checkpoint).
    rules = DEFAULT_METRICS_MAP["regression_rules"]
    if metrics_map is not None and "regression_rules" in metrics_map:
        rules = metrics_map["regression_rules"]
    old_bon = _to_float("bon_accuracy", old_card.get("bon_accuracy", 0.0))
    new_bon = _to_float("bon_accuracy", new_card.get("bon_accuracy", 0.0))
    bon_lift = new_bon - old_bon
    if bon_lift < rules["bon_lift_min"]:
        new = ModelMetrics(
            vgs=new.vgs,
            hack_risk=new.hack_risk,
            calibration=new.calibration,
            ood=new.ood,
            cost=new.cost,
            latency=new.latency,
            regression=True,
        )

    decision = accept_update(tol, old, new)
    return decision, old, new, warnings
