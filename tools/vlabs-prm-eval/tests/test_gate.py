"""Integration tests for ``vlabs-prm-eval gate``.

Exercises the Typer CLI end-to-end against eval-card JSON fixtures.
Each fixture is hand-crafted to trigger exactly one of the seven Lean
``AcceptUpdate`` rejection reasons (plus one accept fixture pair).

The mapping fixture → expected reason mirrors the field order in
``formal/VerifiableLabsFormal/SelfImprovementGate.lean``; if either the
Lean spec or the Python mirror in
``verifiable_labs_envs.formal_spec.gate`` changes, these tests will
catch drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner
from verifiable_labs_envs.formal_spec.gate import (
    REASON_CALIBRATION_DROPPED,
    REASON_COST_EXCEEDED,
    REASON_HACK_RISK_EXCEEDED,
    REASON_LATENCY_EXCEEDED,
    REASON_OOD_DROPPED,
    REASON_REGRESSION_FLAG_SET,
    REASON_VGS_GAIN_BELOW_TAU,
    Tolerances,
)

from vlabs_prm_eval.cli import app
from vlabs_prm_eval.gate import (
    DEFAULT_METRICS_MAP,
    EvalCard,
    card_to_metrics,
    evaluate_gate,
)

FIXTURES = Path(__file__).parent / "fixtures"
OLD = FIXTURES / "card_old.json"


def _runner() -> CliRunner:
    return CliRunner()


# =====================================================================
# Accept path
# =====================================================================
def test_gate_cli_accept():
    """Healthy candidate clears all seven conditions → exit 0, ACCEPT header."""
    result = _runner().invoke(
        app,
        ["gate", "--old", str(OLD), "--new", str(FIXTURES / "card_new_accept.json")],
    )
    assert result.exit_code == 0, result.stdout
    assert "ACCEPT" in result.stdout
    assert "REJECT" not in result.stdout
    assert "Reasons:" not in result.stdout


# =====================================================================
# One reject per AcceptUpdate condition
# =====================================================================
@pytest.mark.parametrize(
    "fixture, expected_reason",
    [
        ("card_reject_vgs.json",        REASON_VGS_GAIN_BELOW_TAU),
        ("card_reject_hack.json",       REASON_HACK_RISK_EXCEEDED),
        ("card_reject_calib.json",      REASON_CALIBRATION_DROPPED),
        ("card_reject_ood.json",        REASON_OOD_DROPPED),
        ("card_reject_cost.json",       REASON_COST_EXCEEDED),
        ("card_reject_latency.json",    REASON_LATENCY_EXCEEDED),
        ("card_reject_regression.json", REASON_REGRESSION_FLAG_SET),
    ],
)
def test_gate_cli_rejects(fixture: str, expected_reason: str):
    """Each fixture triggers its named Lean ``AcceptUpdate`` reason."""
    result = _runner().invoke(
        app,
        ["gate", "--old", str(OLD), "--new", str(FIXTURES / fixture)],
    )
    assert result.exit_code == 1, (fixture, result.stdout)
    assert "REJECT" in result.stdout
    assert expected_reason in result.stdout, (
        f"expected reason {expected_reason!r} not surfaced for {fixture}; "
        f"stdout was:\n{result.stdout}"
    )


# =====================================================================
# Schema bridge — eval-card → ModelMetrics
# =====================================================================
def test_card_to_metrics_default_mapping():
    """Spot-check the documented default weights/aggregations."""
    card = EvalCard.load(FIXTURES / "card_old.json")
    metrics, report = card_to_metrics(card)
    # vgs = 0.4*0.75 + 0.3*0.70 + 0.3*0.65 = 0.705
    assert metrics.vgs == pytest.approx(0.705, abs=1e-9)
    assert metrics.hack_risk == pytest.approx(0.10, abs=1e-9)
    assert metrics.calibration == pytest.approx(0.90, abs=1e-9)
    assert metrics.ood == pytest.approx(0.70, abs=1e-9)
    assert metrics.cost == 1.0 and metrics.latency == 1.0
    assert metrics.regression is False
    assert report.warnings == ()


def test_card_to_metrics_missing_invariance_warns():
    """Missing ``invariance_violation_rate`` defaults hack_risk=0.0 with a warning."""
    import json
    import tempfile

    raw = json.loads((FIXTURES / "card_old.json").read_text())
    raw.pop("invariance_violation_rate")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(raw, f)
        path = Path(f.name)
    try:
        metrics, report = card_to_metrics(EvalCard.load(path))
        assert metrics.hack_risk == 0.0
        assert any("invariance_violation_rate" in w for w in report.warnings)
    finally:
        path.unlink()


def test_evaluate_gate_returns_decision_old_new_warnings():
    """Direct API surface used by the CLI."""
    decision, old_m, new_m, warnings = evaluate_gate(
        old_card_path=OLD,
        new_card_path=FIXTURES / "card_new_accept.json",
        tol=Tolerances(),
    )
    assert decision.accepted is True
    assert decision.reasons == ()
    assert new_m.vgs > old_m.vgs
    assert warnings == []


# =====================================================================
# --metrics-map override
# =====================================================================
def test_metrics_map_override_changes_weights(tmp_path):
    """A custom ``vgs_weights`` map flips an accept fixture into a reject
    by making the BoN component dominate (where the new card improves
    less than tau under the new weighting)."""
    import json

    # Re-weight so processbench_overall is *almost* the entire VGS and
    # the candidate's tiny processbench gain (+0.05) still clears tau,
    # but tighten tau so the gain falls below it.
    override = {
        "vgs_weights": {
            "processbench_overall": 0.10,
            "mean_held_out_env_metric": 0.10,
            "bon_accuracy": 0.80,
        },
        "regression_rules": DEFAULT_METRICS_MAP["regression_rules"],
    }
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(override))

    # bon_lift = 0.72 - 0.65 = 0.07; weighted gain ≈ 0.056 + 0.005 + 0.005 = 0.066
    # That still exceeds tau=0.01. So bump tau above the achievable gain
    # to force the reject.
    result = _runner().invoke(
        app,
        [
            "gate",
            "--old", str(OLD),
            "--new", str(FIXTURES / "card_new_accept.json"),
            "--metrics-map", str(map_path),
            "--tau", "0.10",
        ],
    )
    assert result.exit_code == 1
    assert REASON_VGS_GAIN_BELOW_TAU in result.stdout


# =====================================================================
# Exit-code contract (sanity ladder for CI wiring)
# =====================================================================
def test_gate_exit_zero_on_accept():
    result = _runner().invoke(
        app,
        ["gate", "--old", str(OLD), "--new", str(FIXTURES / "card_new_accept.json")],
    )
    assert result.exit_code == 0


def test_gate_exit_one_on_reject():
    result = _runner().invoke(
        app,
        ["gate", "--old", str(OLD), "--new", str(FIXTURES / "card_reject_vgs.json")],
    )
    assert result.exit_code == 1
