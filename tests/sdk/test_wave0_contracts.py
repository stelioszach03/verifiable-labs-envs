"""Wave-0 contract-locking tests — the SDK-safe interfaces every engine builds
against. These freeze the shape so parallel sub-agents stay connectable."""

from __future__ import annotations

import pytest

from verifiable_labs_envs.providers import (
    AuthMode,
    DummyProvider,
    ModelRequest,
    ProviderConfig,
)
from verifiable_labs_envs.run_config import Mode, RunConfig, default_config
from verifiable_labs_envs.schemas import (
    FORMAL_CLAIM,
    AssuranceCardV2,
    EvaluationContract,
    GateOutcome,
    RiskTier,
    RunContext,
    ScoreSet,
    Split,
    SplitPolicyError,
    TransferMetrics,
    validate_split_policy,
)
from verifiable_labs_envs.schemas.assurance_card import FORBIDDEN_CLAIMS


# ── run config: privacy-preserving defaults ──
def test_default_config_is_privacy_preserving() -> None:
    c = default_config()
    assert c.mode is Mode.EVALUATE_ONLY
    assert c.enable_improvement_suggestions is False
    assert c.enable_candidate_config is False
    assert c.enable_substrate_records is False
    assert c.allow_future_training_use is False
    assert c.public_export is False
    assert c.human_review_required is True
    assert c.dry_run is True


def test_run_config_mode_predicates() -> None:
    assert RunConfig(mode=Mode.EVALUATE_ONLY).does_evaluate
    assert not RunConfig(mode=Mode.EVALUATE_ONLY).does_gate
    assert RunConfig(mode=Mode.GATE_ONLY).does_gate
    assert RunConfig(mode=Mode.IMPROVE_AND_GATE).does_improve
    assert RunConfig(mode=Mode.SUBSTRATE).does_substrate


def test_candidate_config_requires_suggestions() -> None:
    with pytest.raises(ValueError):
        RunConfig(enable_candidate_config=True, enable_improvement_suggestions=False)


def test_run_config_roundtrip() -> None:
    c = RunConfig(mode=Mode.SUBSTRATE, enable_substrate_records=True)
    assert RunConfig.from_dict(c.to_dict()) == c


# ── provider interface + dummy ──
def test_dummy_provider_is_deterministic_and_local() -> None:
    p = DummyProvider()
    req = ModelRequest(prompt="hello agent")
    r1, r2 = p.dry_run(req), p.dry_run(req)
    assert r1.text == r2.text
    assert r1.is_dry_run is True
    assert p.run(req).is_dry_run is False
    assert p.run(req).text == r1.text  # run() is also local/deterministic
    est = p.estimate_cost(req)
    assert est.usd >= 0 and est.tokens_input > 0


def test_provider_config_never_holds_secret_and_redacts_endpoint() -> None:
    cfg = ProviderConfig(
        provider_name="openai",
        model_name="gpt-x",
        auth_mode=AuthMode.CUSTOMER_BYOK,
        endpoint_url="https://api.example.com/v1",
    )
    red = cfg.redacted()
    assert red["endpoint_url"] == "***"
    assert "api_key" not in red and "secret" not in red


def test_provider_config_validates() -> None:
    with pytest.raises(ValueError):
        ProviderConfig(provider_name="", model_name="m")
    with pytest.raises(ValueError):
        ProviderConfig(provider_name="p", model_name="m", temperature=9.0)


# ── split policy (mirrors ContaminationSplits.lean) ──
def test_hidden_eval_never_trainable_or_public() -> None:
    with pytest.raises(SplitPolicyError):
        validate_split_policy(Split.HIDDEN_EVAL, train_allowed=True, public_release_allowed=False)
    with pytest.raises(SplitPolicyError):
        validate_split_policy(Split.HIDDEN_EVAL, train_allowed=False, public_release_allowed=True)
    # valid hidden-eval config raises nothing
    validate_split_policy(Split.HIDDEN_EVAL, train_allowed=False, public_release_allowed=False)


# ── contract: hashes sources, never raw text ──
def test_contract_from_sources_hashes_inputs() -> None:
    c = EvaluationContract.from_sources(
        contract_id="ct_1",
        capability="refunds",
        domain="support",
        user_goal="issue refunds safely",
        sources=["secret customer doc text"],
        risk_tier=RiskTier.HIGH,
    )
    assert c.source_hashes and "secret customer doc text" not in c.source_hashes[0]
    assert len(c.source_hashes[0]) == 64  # sha256 hex
    assert c.to_dict()["risk_tier"] == "high"


# ── results + transfer ──
def test_scoreset_bounds_and_transfer() -> None:
    s = ScoreSet(
        public_score=0.9,
        hidden_score=0.6,
        ood_score=0.5,
        adversarial_score=0.4,
        dcr=0.2,
        hack_risk=0.05,
        calibration=0.9,
    )
    tm = TransferMetrics.from_scores(s)
    assert abs(tm.public_to_hidden_gap - 0.3) < 1e-9
    assert tm.clean_transfer_score == pytest.approx(0.6 * 0.8)
    with pytest.raises(ValueError):
        ScoreSet(
            public_score=1.2,
            hidden_score=0.6,
            ood_score=0.5,
            adversarial_score=0.4,
            dcr=0.2,
            hack_risk=0.0,
            calibration=0.9,
        )


def test_gate_outcome_labels() -> None:
    g = GateOutcome(label=GateOutcome.ACCEPT)
    assert g.accepted
    with pytest.raises(ValueError):
        GateOutcome(label="MAYBE")


def test_run_context_serializes() -> None:
    rc = RunContext(run_id="run_1", mode="evaluate_only")
    assert rc.to_dict()["run_id"] == "run_1"


# ── assurance card: honesty rule baked in ──
def test_assurance_card_locks_formal_claim() -> None:
    card = AssuranceCardV2(
        card_version="v2",
        run_id="run_1",
        agent_id="a",
        baseline_id=None,
        candidate_id=None,
        decision="ACCEPT",
    )
    assert card.formal_claim == FORMAL_CLAIM
    assert "machine-verified in Lean 4" in card.formal_claim
    # forbidden phrasings must never be the claim
    low = card.formal_claim.lower()
    for bad in FORBIDDEN_CLAIMS:
        assert bad not in low
    with pytest.raises(ValueError):
        AssuranceCardV2(
            card_version="v2",
            run_id="r",
            agent_id=None,
            baseline_id=None,
            candidate_id=None,
            decision="ACCEPT",
            formal_claim="formally verified API",  # not a valid claim; must raise
        )
