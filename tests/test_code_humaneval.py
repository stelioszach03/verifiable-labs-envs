"""Tests for the ``code-humaneval`` env (Phase 24.B).

Coverage:
- Procedural template lattice — 12 templates, deterministic per seed,
  EFFECTIVE_INSTANCES > 1e15.
- Reward kernel — format/parse short-circuits, gold-solution full
  pass-rate, broken solution drops pass_rate below 1.
- Adapter — JSON envelope parse, fenced-block tolerance, malformed
  inputs degrade to empty-code prediction.
- Env class contract — score returns the canonical
  ``{reward, components, meta}`` dict; ``meta`` carries the conformal
  ``covered`` flag.
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

from verifiable_labs_envs.envs.code_humaneval import (
    _TEMPLATES,
    DEFAULT_HYPERPARAMS,
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    SYSTEM_PROMPT,
    CodeHumanevalEnv,
    CodePrediction,
    baseline_predict,
    build_user_prompt,
    compute_reward,
    generate_instance,
    generate_problem,
    load_environment,
    parse_response,
    score_components,
)
from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    _unshare_available as _sandbox_capable,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not _sandbox_capable(),
    reason=(
        "code-humaneval / code-mini-repo scoring requires the Linux "
        "sandbox primitive (unshare -rn). GitHub-hosted ubuntu-latest "
        "runners ship the binary but kernel rejects uid_map writes."
    ),
)


# ── Catalogue / metadata ──────────────────────────────────────────────


def test_name_is_kebab_case() -> None:
    assert NAME == "code-humaneval"
    assert "_" not in NAME


def test_effective_instances_above_procedural_threshold() -> None:
    assert EFFECTIVE_INSTANCES > 1e15


def test_default_weights_sum_to_one() -> None:
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_default_weights_match_plan_d7c() -> None:
    """D7-C ruling: 0.10 format + 0.20 parse + 0.70 pass_rate."""
    assert DEFAULT_WEIGHTS["format_valid"] == pytest.approx(0.10)
    assert DEFAULT_WEIGHTS["parse_valid"] == pytest.approx(0.20)
    assert DEFAULT_WEIGHTS["pass_rate"] == pytest.approx(0.70)


def test_default_hyperparams_carry_alpha_and_timeout() -> None:
    assert "alpha" in DEFAULT_HYPERPARAMS
    assert 0.0 < DEFAULT_HYPERPARAMS["alpha"] < 1.0
    assert "sandbox_timeout_s" in DEFAULT_HYPERPARAMS
    assert DEFAULT_HYPERPARAMS["sandbox_timeout_s"] > 0


# ── Procedural lattice ────────────────────────────────────────────────


def test_template_count_matches_plan() -> None:
    """PHASE_24_PLAN.md §8.1 locks 12 templates."""
    assert len(_TEMPLATES) == 12


def test_generate_problem_is_deterministic() -> None:
    a = generate_problem(seed=42)
    b = generate_problem(seed=42)
    assert a == b


def test_generate_problem_varies_with_seed() -> None:
    """Different seeds → different problems with high probability."""
    seen = {generate_problem(s)["template_name"] for s in range(50)}
    # All 12 templates should appear over 50 seeds.
    assert len(seen) >= 8


def test_generate_instance_carries_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    assert inst.gold_solution
    assert inst.hidden_tests
    assert inst.visible_tests


def test_as_inputs_excludes_oracle_fields() -> None:
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_solution" not in inputs
    assert "hidden_tests" not in inputs
    assert "prompt" in inputs
    assert "visible_tests" in inputs


def test_prompt_includes_signature_and_visible_tests() -> None:
    inst = generate_instance(seed=0)
    text = inst.prompt
    assert inst.function_signature in text
    for vt in inst.visible_tests:
        assert vt in text


# ── Reward components — pure (no sandbox) ─────────────────────────────


def _toy_prediction(code: str) -> CodePrediction:
    return CodePrediction(
        code=code,
        raw=json.dumps({"code": code, "confidence": 0.5}),
        confidence=0.5,
    )


def test_score_components_bad_json_short_circuits() -> None:
    inst = generate_instance(seed=0)
    pred = CodePrediction(code="", raw="not json", confidence=0.0)
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 0.0
    assert components["parse_valid"] == 0.0
    assert components["pass_rate"] == 0.0


def test_score_components_uncompileable_short_circuits() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction("def broken(:\n  pass")  # syntax error
    components = score_components(pred, inst, timeout_s=2.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
    assert components["pass_rate"] == 0.0


def test_score_components_gold_solution_passes_all_tests() -> None:
    """The procedural gold solution must clear every visible + hidden case."""
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_solution)
    components = score_components(pred, inst, timeout_s=10.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    assert components["pass_rate"] == pytest.approx(1.0), (
        f"gold did not pass: stdout sample={inst.template_name!r}"
    )


def test_score_components_returns_zero_solution_below_one() -> None:
    """A `def f(*a, **k): return 0` stub fails some hidden tests."""
    inst = generate_instance(seed=0)
    stub = textwrap.dedent(
        """
        def __getattr__(name):
            def _f(*a, **k):
                return 0
            return _f
        """
    ).strip()
    pred = _toy_prediction(stub)
    components = score_components(pred, inst, timeout_s=10.0)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 1.0
    # Some templates may pass with 0 (e.g. all-zero edge case), but the
    # full hidden-test suite spans positive expectations too — pass_rate
    # cannot be 1.0 across the board.
    assert components["pass_rate"] < 1.0


# ── Reward aggregation ───────────────────────────────────────────────


def test_compute_reward_in_unit_range() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_solution)
    out = compute_reward(prediction=pred, instance=inst, timeout_s=10.0)
    assert 0.0 <= out["reward"] <= 1.0
    assert out["reward"] == pytest.approx(1.0, rel=0.01)


def test_compute_reward_reports_template_name_in_meta() -> None:
    inst = generate_instance(seed=3)
    pred = baseline_predict(inst)
    out = compute_reward(prediction=pred, instance=inst, timeout_s=2.0)
    assert out["meta"]["template"] == inst.template_name


def test_compute_reward_with_conformal_emits_covered_flag() -> None:
    inst = generate_instance(seed=0)
    pred = _toy_prediction(inst.gold_solution)
    out = compute_reward(
        prediction=pred,
        instance=inst,
        timeout_s=10.0,
        conformal_quantile=0.5,
    )
    meta = out["meta"]
    assert "covered" in meta
    assert isinstance(meta["covered"], bool)
    assert "residual" in meta
    assert 0.0 <= meta["residual"] <= 1.0


# ── Env class ────────────────────────────────────────────────────────


def test_load_environment_returns_env_instance() -> None:
    env = load_environment(calibration_quantile=0.5)
    assert isinstance(env, CodeHumanevalEnv)
    assert env.name == NAME


def test_env_score_returns_canonical_dict() -> None:
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    pred = baseline_predict(inst)
    out = env.score(pred, inst)
    assert "reward" in out
    assert "components" in out
    assert "meta" in out
    assert "covered" in out["meta"]
    for k, v in out["components"].items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"


def test_env_run_baseline_finite_reward() -> None:
    env = load_environment(calibration_quantile=0.5)
    out = env.run_baseline(seed=0)
    assert isinstance(out["reward"], float)
    assert 0.0 <= out["reward"] <= 1.0


# ── Adapter ──────────────────────────────────────────────────────────


def test_system_prompt_documents_json_envelope() -> None:
    assert "code" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


def test_build_user_prompt_includes_signature() -> None:
    inst = generate_instance(seed=0)
    user_prompt = build_user_prompt(inst)
    assert inst.function_signature in user_prompt
    assert "OUTPUT SCHEMA" in user_prompt


def test_parse_response_handles_clean_json() -> None:
    inst = generate_instance(seed=0)
    payload = json.dumps({"code": "def f(): return 1", "confidence": 0.7})
    pred = parse_response(payload, inst)
    assert pred.code == "def f(): return 1"
    assert pred.confidence == pytest.approx(0.7)


def test_parse_response_handles_fenced_json() -> None:
    inst = generate_instance(seed=0)
    text = (
        "Sure, here's the code:\n"
        "```json\n"
        '{"code": "def f(): return 2", "confidence": 0.4}\n'
        "```"
    )
    pred = parse_response(text, inst)
    assert pred.code == "def f(): return 2"


def test_parse_response_returns_empty_on_garbage() -> None:
    inst = generate_instance(seed=0)
    pred = parse_response("totally unrelated prose without a JSON block", inst)
    assert pred.code == ""
    assert pred.confidence == 0.0


def test_baseline_predict_is_empty() -> None:
    inst = generate_instance(seed=0)
    pred = baseline_predict(inst)
    assert pred.code == ""
    assert pred.confidence == 0.0


def test_baseline_scores_zero_reward() -> None:
    """Empty prediction → zero on all components → reward = 0."""
    env = load_environment(calibration_quantile=0.5)
    inst = env.generate_instance(seed=0)
    out = env.score(baseline_predict(inst), inst)
    assert out["reward"] == 0.0
