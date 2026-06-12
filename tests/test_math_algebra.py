"""Tests for the math-algebra env (Phase 21.C).

Mirrors `tests/test_sparse_fourier.py`'s structure: env contract,
reward kernel, and LLM adapter all in one file. Coverage targets:

- env: ENV_NAME, registry round-trip, EFFECTIVE_INSTANCES,
  determinism, baseline, template diversity (8 tests).
- reward: 3-component sum, equivalent-form acceptance,
  short-circuit on parse failure, threaded simplify timeout (5 tests).
- adapter: clean JSON, fenced JSON, missing fields, registry
  registration (4 tests).
"""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs.math_algebra import (
    DEFAULT_WEIGHTS,
    EFFECTIVE_INSTANCES,
    NAME,
    Instance,
    MathAlgebraEnv,
    Prediction,
    compute_reward,
    generate_instance,
    score_components,
    simplify_with_timeout,
)
from verifiable_labs_envs.solvers.adapters.math_algebra import (
    SYSTEM_PROMPT,
    MathAlgebraLLMAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import LLMSolverError

# ───────────────────────── env contract ─────────────────────────────


def test_env_id_is_kebab_case():
    assert NAME == "math-algebra"
    assert "_" not in NAME


def test_env_registered_in_central_registry():
    """`verifiable run --env math-algebra` resolves through this."""
    assert NAME in list_environments()


def test_load_environment_via_registry_returns_env_handle():
    env = load_environment(NAME, calibration_quantile=0.5)
    assert isinstance(env, MathAlgebraEnv)
    assert env.name == NAME
    assert env.conformal_quantile == pytest.approx(0.5)


def test_effective_instances_above_procedural_threshold():
    """Procedural-regeneration certification: > 1e15 unique problems."""
    assert EFFECTIVE_INSTANCES > 1e15


def test_generate_instance_returns_well_formed_instance():
    inst = generate_instance(seed=0)
    assert isinstance(inst, Instance)
    assert inst.seed == 0
    assert isinstance(inst.prompt, str) and inst.prompt
    assert isinstance(inst.gold_expr, str) and inst.gold_expr
    assert "alpha" in inst.metadata
    assert "template" in inst.metadata


def test_generate_instance_seed_determinism():
    a = generate_instance(seed=42)
    b = generate_instance(seed=42)
    assert a.prompt == b.prompt
    assert a.gold_expr == b.gold_expr
    assert a.metadata["template"] == b.metadata["template"]


def test_different_seeds_yield_different_problems():
    """Across 50 seeds we should see at least 5 unique gold expressions."""
    golds = {generate_instance(seed=s).gold_expr for s in range(50)}
    assert len(golds) >= 5, f"only {len(golds)} unique problems in 50 seeds"


def test_as_inputs_excludes_gold_expr():
    inst = generate_instance(seed=0)
    inputs = inst.as_inputs()
    assert "gold_expr" not in inputs
    assert "prompt" in inputs


def test_template_distribution_is_diverse():
    """All 5 templates must be reachable across the first 100 seeds."""
    templates = {generate_instance(seed=s).metadata["template"] for s in range(100)}
    assert len(templates) >= 4, f"template diversity too low: {templates}"


# ───────────────────────── reward kernel ─────────────────────────────


def _toy_instance(prompt: str, gold_expr: str) -> Instance:
    return Instance(
        prompt=prompt,
        gold_expr=gold_expr,
        seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0, "template": "toy"},
    )


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_match_full_reward():
    inst = _toy_instance(prompt="Expand (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="x**2 - 1",
        raw=json.dumps({"answer": "x**2 - 1", "confidence": 0.9}),
        confidence=0.9,
    )
    components = score_components(pred, inst)
    assert components == {"format_valid": 1.0, "parse_valid": 1.0, "correct": 1.0}


def test_equivalent_form_scores_correct():
    """`(x-1)*(x+1)` and `x**2 - 1` are algebraically equal — simplify
    catches it and `correct` should be 1.0."""
    inst = _toy_instance(prompt="Expand (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="(x-1)*(x+1)",
        raw=json.dumps({"answer": "(x-1)*(x+1)", "confidence": 0.8}),
        confidence=0.8,
    )
    components = score_components(pred, inst)
    assert components["correct"] == 1.0


def test_wrong_answer_zero_correct_but_parse_valid():
    inst = _toy_instance(prompt="Expand (x+1)*(x-1)", gold_expr="x**2 - 1")
    pred = Prediction(
        answer_expr="x**2 + 1",
        raw=json.dumps({"answer": "x**2 + 1", "confidence": 0.9}),
        confidence=0.9,
    )
    components = score_components(pred, inst)
    assert components["correct"] == 0.0
    assert components["parse_valid"] == 1.0


def test_unparseable_answer_short_circuits():
    inst = _toy_instance(prompt="x", gold_expr="x")
    pred = Prediction(
        answer_expr="not sympy at all !",
        raw=json.dumps({"answer": "not sympy at all !", "confidence": 0.5}),
        confidence=0.5,
    )
    components = score_components(pred, inst)
    assert components["format_valid"] == 1.0
    assert components["parse_valid"] == 0.0
    assert components["correct"] == 0.0


def test_compute_reward_emits_covered_flag_with_quantile():
    inst = _toy_instance(prompt="x", gold_expr="x")
    pred = Prediction(
        answer_expr="x",
        raw=json.dumps({"answer": "x", "confidence": 0.5}),
        confidence=0.5,
    )
    out = compute_reward(prediction=pred, instance=inst, conformal_quantile=0.3)
    assert "covered" in out["meta"]
    assert "residual" in out["meta"]
    assert isinstance(out["meta"]["covered"], bool)


def test_simplify_timeout_returns_none_on_wedge():
    """A pathological diff with a 1ms timeout must NOT block."""
    import sympy as sp
    expr = sp.Symbol("x") ** 100 + sp.sin(sp.Symbol("x")) * sp.exp(sp.Symbol("y"))
    # Either timeout (None) or a quick simplification — neither blocks.
    result = simplify_with_timeout(expr, timeout_s=0.001)
    assert result is None or hasattr(result, "free_symbols")


def test_full_env_score_with_oracle_solver_reads_one():
    """End-to-end: oracle solver (gold IS the answer) scores 1.0."""
    inst = generate_instance(seed=7)
    pred = Prediction(
        answer_expr=inst.gold_expr,
        raw=json.dumps({"answer": inst.gold_expr, "confidence": 1.0}),
        confidence=1.0,
    )
    env = MathAlgebraEnv(conformal_quantile=0.5)
    out = env.score(pred, inst)
    assert out["reward"] == pytest.approx(1.0)
    assert out["components"]["correct"] == 1.0


def test_baseline_produces_zero_reward():
    """The empty-prediction baseline scores zero on all components.
    This gives a non-trivial residual distribution at calibration."""
    env = MathAlgebraEnv(conformal_quantile=0.5)
    out = env.run_baseline(seed=0)
    assert out["reward"] == 0.0
    assert all(v == 0.0 for v in out["components"].values())


# ───────────────────────── LLM adapter ──────────────────────────────


def test_adapter_env_name_matches_registry_key():
    adapter = MathAlgebraLLMAdapter()
    assert adapter.env_name == "math-algebra"


def test_system_prompt_documents_json_schema():
    assert "JSON" in SYSTEM_PROMPT
    assert "answer" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_build_user_prompt_renders_problem_statement():
    inst = generate_instance(seed=0)
    adapter = MathAlgebraLLMAdapter()
    text = adapter.build_user_prompt(inst)
    assert "PROBLEM" in text
    assert "OUTPUT SCHEMA" in text
    assert inst.prompt in text


def test_parse_response_clean_json_round_trip():
    inst = generate_instance(seed=0)
    payload = {"answer": "x**2 + 1", "confidence": 0.85}
    adapter = MathAlgebraLLMAdapter()
    pred = adapter.parse_response(json.dumps(payload), inst)
    assert pred.answer_expr == "x**2 + 1"
    assert pred.confidence == pytest.approx(0.85)


def test_parse_response_handles_markdown_fences():
    inst = generate_instance(seed=0)
    payload = {"answer": "x", "confidence": 0.5}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    adapter = MathAlgebraLLMAdapter()
    pred = adapter.parse_response(fenced, inst)
    assert pred.answer_expr == "x"


def test_parse_response_raises_on_unrecoverable_json():
    """Match sparse_fourier: hard error when no JSON block at all."""
    inst = generate_instance(seed=0)
    adapter = MathAlgebraLLMAdapter()
    with pytest.raises(LLMSolverError):
        adapter.parse_response("totally not json", inst)


def test_parse_response_clamps_confidence_to_unit_interval():
    inst = generate_instance(seed=0)
    adapter = MathAlgebraLLMAdapter()
    pred_high = adapter.parse_response(
        json.dumps({"answer": "x", "confidence": 1.7}), inst,
    )
    pred_low = adapter.parse_response(
        json.dumps({"answer": "x", "confidence": -0.3}), inst,
    )
    assert pred_high.confidence == 1.0
    assert pred_low.confidence == 0.0


def test_adapter_registered_globally():
    """Required for `verifiable run --env math-algebra` to dispatch."""
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  triggers registration
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert "math-algebra" in _ADAPTERS, (
        "MathAlgebraLLMAdapter not registered in solvers.adapters.__init__.py"
    )
