"""Eval-card → ``CleanMetrics`` mapping + clean-gate evaluator.

The eight-condition ``CleanAcceptUpdate`` predicate itself lives in
``verifiable_labs_envs.formal_spec.clean_promotion_gate`` (machine-verified in
Lean 4, namespace ``Verifiable.CleanPromotionGate``). This module only handles
the schema bridge from a process-reward-model eval card (JSON dict) to
``CleanMetrics`` and the user-facing rendering — mirroring ``gate.py``.

Eval-card schema (canonical fields, all optional unless noted):

    {
      "model_id":                  "<str, required>",
      "vgs" | "composite":         <float in [0,1]>,        # raw_vgs
      "contamination_risk" | "dcr":<float in [0,1]>,        # dcr (else 0.0, WARN)
      "clean_vgs":                 <float>,                  # else computed
      "public_score":              <float in [0,1]>,
      "hidden_score":              <float in [0,1]>,
      "ood_score":                 <float in [0,1]>,
      "hack_risk":                 <float>,                  # else 0.0, WARN
      "calibration":               <float in [0,1]>,         # else 1.0, WARN
      "cost":                      <float, $ per audit>,
      "latency":                   <float, seconds per audit>,
      "regression":                <bool>                    # else derived
    }

The mapping is documented below and adjustable via the CLI ``--metrics-map``
flag (a JSON object overriding the default field names / fallbacks).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiable_labs_envs.formal_spec.clean_promotion_gate import (
    CleanGateDecision,
    CleanMetrics,
    CleanTolerances,
    accept_clean_update,
)
from verifiable_labs_envs.formal_spec.clean_vgs import clean_vgs as compute_clean_vgs

# ---------------------------------------------------------------------
# Default eval-card → CleanMetrics field aliases + fallbacks
# ---------------------------------------------------------------------
DEFAULT_METRICS_MAP: dict[str, Any] = {
    # Ordered alias lists: first present key wins.
    "raw_vgs_keys": ["vgs", "composite"],
    "dcr_keys": ["contamination_risk", "dcr"],
    "clean_vgs_keys": ["clean_vgs"],
    "public_score_keys": ["public_score"],
    "hidden_score_keys": ["hidden_score"],
    "ood_score_keys": ["ood_score"],
    "hack_risk_keys": ["hack_risk"],
    "calibration_keys": ["calibration"],
    "cost_keys": ["cost"],
    "latency_keys": ["latency"],
    "regression_keys": ["regression"],
    # Fallback values when absent.
    "dcr_default": 0.0,
    "hack_risk_default": 0.0,
    "calibration_default": 1.0,
    "cost_default": 0.0,
    "latency_default": 0.0,
}


@dataclass(frozen=True)
class CleanEvalCard:
    """Minimal eval-card view used by the clean gate. Extra fields tolerated."""

    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> CleanEvalCard:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a JSON object, got {type(data).__name__}"
            )
        return cls(raw=data)

    def first(self, keys: list[str]) -> tuple[str | None, Any]:
        for k in keys:
            if k in self.raw and self.raw[k] is not None:
                return k, self.raw[k]
        return None, None


@dataclass(frozen=True)
class CleanMappingReport:
    """Diagnostic for the eval-card → CleanMetrics bridge."""

    warnings: tuple[str, ...] = field(default_factory=tuple)


def _to_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected number, got {value!r}")
    return float(value)


def card_to_clean_metrics(
    card: CleanEvalCard,
    beta: float,
    metrics_map: Mapping[str, Any] | None = None,
    *,
    cost_override: float | None = None,
    latency_override: float | None = None,
) -> tuple[CleanMetrics, CleanMappingReport]:
    """Map an eval card onto ``CleanMetrics``.

    Returns the metrics plus a report of any warnings (e.g. a missing
    contamination-risk / hack-risk / calibration field that fell back to a
    default). Raises ``ValueError`` on hard schema errors.

    The contamination-adjusted ``clean_vgs`` is taken from the card when
    present; otherwise it is computed from ``raw_vgs``, ``dcr`` and ``beta`` via
    the machine-verified formula ``clean_vgs = raw_vgs*(1-dcr) - beta*dcr``.
    """
    m = dict(DEFAULT_METRICS_MAP)
    if metrics_map is not None:
        m = {**m, **metrics_map}

    warnings: list[str] = []

    # ---- raw_vgs ----
    key, val = card.first(m["raw_vgs_keys"])
    if key is None:
        raise ValueError(
            f"card has none of {m['raw_vgs_keys']!r}; cannot derive raw_vgs."
        )
    raw_vgs = _to_float("raw_vgs", val)

    # ---- dcr ----
    key, val = card.first(m["dcr_keys"])
    if key is None:
        warnings.append(
            f"card has no contamination-risk field {m['dcr_keys']!r}; "
            f"using dcr={m['dcr_default']}. The clean gate cannot penalise "
            "contamination without it."
        )
        dcr = float(m["dcr_default"])
    else:
        dcr = _to_float("dcr", val)

    # ---- clean_vgs (card value, else computed) ----
    key, val = card.first(m["clean_vgs_keys"])
    if key is None:
        clean_vgs_value = compute_clean_vgs(raw_vgs, dcr, beta)
    else:
        clean_vgs_value = _to_float("clean_vgs", val)

    # ---- public / hidden / ood scores ----
    _, pub = card.first(m["public_score_keys"])
    public_score = _to_float("public_score", pub) if pub is not None else 0.0
    _, hid = card.first(m["hidden_score_keys"])
    hidden_score = _to_float("hidden_score", hid) if hid is not None else 0.0
    _, ood = card.first(m["ood_score_keys"])
    ood_score = _to_float("ood_score", ood) if ood is not None else 0.0

    # ---- hack_risk ----
    key, val = card.first(m["hack_risk_keys"])
    if key is None:
        warnings.append(
            f"card has no hack_risk field {m['hack_risk_keys']!r}; using "
            f"hack_risk={m['hack_risk_default']}. Run the invariance harness to "
            "populate it; the gate cannot detect shortcuts without it."
        )
        hack_risk = float(m["hack_risk_default"])
    else:
        hack_risk = _to_float("hack_risk", val)

    # ---- calibration ----
    key, val = card.first(m["calibration_keys"])
    if key is None:
        warnings.append(
            f"card has no calibration field {m['calibration_keys']!r}; using "
            f"calibration={m['calibration_default']} (assumed nominal)."
        )
        calibration = float(m["calibration_default"])
    else:
        calibration = _to_float("calibration", val)

    # ---- cost / latency ----
    if cost_override is not None:
        cost = cost_override
    else:
        key, val = card.first(m["cost_keys"])
        cost = _to_float("cost", val) if key is not None else float(m["cost_default"])
    if latency_override is not None:
        latency = latency_override
    else:
        key, val = card.first(m["latency_keys"])
        latency = (
            _to_float("latency", val) if key is not None else float(m["latency_default"])
        )

    # ---- regression (card value, else derived from contamination/calibration) ----
    key, val = card.first(m["regression_keys"])
    if key is not None:
        if not isinstance(val, bool):
            raise ValueError(f"regression: expected bool, got {val!r}")
        regression = val
    else:
        # Derived: a fully-contaminated or clearly mis-calibrated candidate is a
        # hard regression. Honest, conservative default.
        regression = bool(dcr >= 1.0 or calibration <= 0.0)

    return (
        CleanMetrics(
            raw_vgs=raw_vgs,
            dcr=dcr,
            clean_vgs=clean_vgs_value,
            public_score=public_score,
            hidden_score=hidden_score,
            ood_score=ood_score,
            hack_risk=hack_risk,
            calibration=calibration,
            cost=cost,
            latency=latency,
            regression=regression,
        ),
        CleanMappingReport(warnings=tuple(warnings)),
    )


def evaluate_clean_gate(
    old_card_path: str | Path,
    new_card_path: str | Path,
    tol: CleanTolerances,
    beta: float,
    metrics_map_path: str | Path | None = None,
    *,
    cost_old: float | None = None,
    cost_new: float | None = None,
    latency_old: float | None = None,
    latency_new: float | None = None,
) -> tuple[CleanGateDecision, CleanMetrics, CleanMetrics, list[str]]:
    """Load two cards, map to clean metrics, evaluate the clean gate.

    Returns ``(decision, old_metrics, new_metrics, warnings)``.
    """
    metrics_map: Mapping[str, Any] | None = None
    if metrics_map_path is not None:
        with open(metrics_map_path, encoding="utf-8") as f:
            metrics_map = json.load(f)

    old_card = CleanEvalCard.load(old_card_path)
    new_card = CleanEvalCard.load(new_card_path)

    old, report_old = card_to_clean_metrics(
        old_card, beta, metrics_map, cost_override=cost_old, latency_override=latency_old
    )
    new, report_new = card_to_clean_metrics(
        new_card, beta, metrics_map, cost_override=cost_new, latency_override=latency_new
    )

    warnings = [*report_old.warnings, *report_new.warnings]
    decision = accept_clean_update(tol, old, new)
    return decision, old, new, warnings


__all__ = [
    "DEFAULT_METRICS_MAP",
    "CleanEvalCard",
    "CleanMappingReport",
    "card_to_clean_metrics",
    "evaluate_clean_gate",
]
