"""Integration tests for ``vlabs-prm-eval clean-gate``.

Exercises the Typer CLI end-to-end against eval-card JSON fixtures. Each reject
fixture is hand-crafted to trigger exactly one of the eight Lean
``CleanAcceptUpdate`` rejection reasons (plus one accept fixture pair and one
fixture that exercises the missing-field warnings path).

The mapping fixture → expected reason mirrors the field order in
``formal/VerifiableLabsFormal/CleanPromotionGate.lean``; if either the Lean spec
or the Python mirror in ``verifiable_labs_envs.formal_spec.clean_promotion_gate``
changes, these tests catch drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner
from verifiable_labs_envs.formal_spec.clean_promotion_gate import (
    REASON_CALIBRATION_REGRESSED,
    REASON_CLEAN_VGS_NOT_IMPROVED,
    REASON_COST_INCREASED,
    REASON_DCR_INCREASED,
    REASON_HACK_RISK_INCREASED,
    REASON_LATENCY_INCREASED,
    REASON_OOD_REGRESSED,
    REASON_REGRESSION_FLAGGED,
    CleanTolerances,
)

from vlabs_prm_eval.clean_gate import (
    CleanEvalCard,
    card_to_clean_metrics,
    evaluate_clean_gate,
)
from vlabs_prm_eval.cli import app

FIXTURES = Path(__file__).parent / "fixtures" / "clean"
OLD = FIXTURES / "clean_old.json"


def _runner() -> CliRunner:
    return CliRunner()


# =====================================================================
# Accept path
# =====================================================================
def test_clean_gate_cli_accept():
    """Healthy candidate clears all eight conditions → exit 0, ACCEPT header."""
    result = _runner().invoke(
        app,
        ["clean-gate", "--old", str(OLD), "--new", str(FIXTURES / "clean_new_accept.json")],
    )
    assert result.exit_code == 0, result.stdout
    assert "ACCEPT" in result.stdout
    assert "REJECT" not in result.stdout
    assert "Reasons:" not in result.stdout


def test_clean_gate_does_not_remove_gate_command():
    """The original ``gate`` command must still exist alongside ``clean-gate``."""
    result = _runner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "clean-gate" in result.stdout
    assert "gate" in result.stdout


# =====================================================================
# One reject per CleanAcceptUpdate condition
# =====================================================================
@pytest.mark.parametrize(
    "fixture, expected_reason",
    [
        ("clean_reject_clean_vgs.json",  REASON_CLEAN_VGS_NOT_IMPROVED),
        ("clean_reject_hack.json",       REASON_HACK_RISK_INCREASED),
        ("clean_reject_calib.json",      REASON_CALIBRATION_REGRESSED),
        ("clean_reject_ood.json",        REASON_OOD_REGRESSED),
        ("clean_reject_dcr.json",        REASON_DCR_INCREASED),
        ("clean_reject_cost.json",       REASON_COST_INCREASED),
        ("clean_reject_latency.json",    REASON_LATENCY_INCREASED),
        ("clean_reject_regression.json", REASON_REGRESSION_FLAGGED),
    ],
)
def test_clean_gate_cli_rejects(fixture: str, expected_reason: str):
    """Each fixture triggers its named Lean ``CleanAcceptUpdate`` reason."""
    result = _runner().invoke(
        app,
        ["clean-gate", "--old", str(OLD), "--new", str(FIXTURES / fixture)],
    )
    assert result.exit_code == 1, (fixture, result.stdout)
    assert "REJECT" in result.stdout
    assert expected_reason in result.stdout, (
        f"expected reason {expected_reason!r} not surfaced for {fixture}; "
        f"stdout was:\n{result.stdout}"
    )


def test_each_reject_fixture_triggers_exactly_one_reason():
    """Direct API: each reject fixture yields exactly one reason."""
    tol = CleanTolerances()
    cases = {
        "clean_reject_clean_vgs.json":  REASON_CLEAN_VGS_NOT_IMPROVED,
        "clean_reject_hack.json":       REASON_HACK_RISK_INCREASED,
        "clean_reject_calib.json":      REASON_CALIBRATION_REGRESSED,
        "clean_reject_ood.json":        REASON_OOD_REGRESSED,
        "clean_reject_dcr.json":        REASON_DCR_INCREASED,
        "clean_reject_cost.json":       REASON_COST_INCREASED,
        "clean_reject_latency.json":    REASON_LATENCY_INCREASED,
        "clean_reject_regression.json": REASON_REGRESSION_FLAGGED,
    }
    for fixture, reason in cases.items():
        decision, _, _, _ = evaluate_clean_gate(
            OLD, FIXTURES / fixture, tol, beta=0.5
        )
        assert decision.accepted is False, fixture
        assert decision.reasons == (reason,), (fixture, decision.reasons)


# =====================================================================
# Warnings path — card missing dcr / hack_risk / calibration
# =====================================================================
def test_clean_gate_warns_on_missing_fields():
    """A card missing dcr/hack_risk/calibration emits warnings and still maps."""
    decision, _old, _new, warnings = evaluate_clean_gate(
        OLD, FIXTURES / "clean_new_warnings.json", CleanTolerances(), beta=0.5
    )
    joined = " ".join(warnings)
    assert "contamination-risk" in joined
    assert "hack_risk" in joined
    assert "calibration" in joined
    assert len(warnings) == 3
    # The decision still resolves (defaults applied), here an accept.
    assert decision.accepted is True


def test_clean_gate_cli_emits_warnings_to_stderr():
    result = _runner().invoke(
        app,
        ["clean-gate", "--old", str(OLD), "--new", str(FIXTURES / "clean_new_warnings.json")],
    )
    # Warnings go to stderr; mix=False keeps them separate. CliRunner merges by
    # default, so assert the WARNING text appears in the combined output.
    assert "WARNING" in result.output


def test_card_to_clean_metrics_computes_clean_vgs_when_absent():
    """If clean_vgs is absent the bridge computes it via the verified formula."""
    card = CleanEvalCard.load(FIXTURES / "clean_new_warnings.json")
    metrics, report = card_to_clean_metrics(card, beta=0.5)
    # vgs=0.80, dcr default 0.0 -> clean_vgs = 0.80*(1-0) - 0.5*0 = 0.80
    assert metrics.clean_vgs == pytest.approx(0.80, abs=1e-9)
    assert metrics.dcr == 0.0
    assert metrics.hack_risk == 0.0
    assert metrics.calibration == 1.0
    assert len(report.warnings) == 3


# =====================================================================
# Exit-code contract
# =====================================================================
def test_clean_gate_exit_zero_on_accept():
    result = _runner().invoke(
        app,
        ["clean-gate", "--old", str(OLD), "--new", str(FIXTURES / "clean_new_accept.json")],
    )
    assert result.exit_code == 0


def test_clean_gate_exit_one_on_reject():
    result = _runner().invoke(
        app,
        ["clean-gate", "--old", str(OLD), "--new", str(FIXTURES / "clean_reject_dcr.json")],
    )
    assert result.exit_code == 1
