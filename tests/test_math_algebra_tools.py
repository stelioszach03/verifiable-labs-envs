"""Tests for the math-algebra-tools env (Phase 21.E).

Mirror of `tests/test_sparse_fourier_tools.py`'s structure: env
contract, tool dispatch, schema validation, registry registration.
"""
from __future__ import annotations

import json

import pytest

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs.math_algebra import (
    Instance,
    MathAlgebraEnv,
    Prediction,
    generate_instance,
)
from verifiable_labs_envs.envs.math_algebra_tools import (
    DEFAULT_MAX_TOOL_CALLS,
    NAME,
    TOOL_SCHEMAS,
    MathAlgebraToolsEnv,
    dispatch_tool,
)
from verifiable_labs_envs.solvers.adapters.math_algebra_tools import (
    SYSTEM_PROMPT_TOOLS,
    MathAlgebraToolsAdapter,
)

# ───────────────────────── env contract ─────────────────────────────


def test_env_id_is_kebab_case():
    assert NAME == "math-algebra-tools"
    assert "_" not in NAME


def test_env_registered_in_central_registry():
    assert NAME in list_environments()


def test_load_environment_via_registry_returns_subclass():
    env = load_environment(NAME, calibration_quantile=0.5)
    assert isinstance(env, MathAlgebraToolsEnv)
    # Tools env must subclass single-turn (verifier reuse).
    assert isinstance(env, MathAlgebraEnv)
    assert env.name == NAME
    assert env.max_tool_calls == DEFAULT_MAX_TOOL_CALLS


def test_max_tool_calls_negative_rejected():
    with pytest.raises(ValueError, match="max_tool_calls"):
        MathAlgebraToolsEnv(conformal_quantile=0.5, max_tool_calls=-1)


def test_max_tool_calls_zero_allowed():
    """Zero tool calls = direct-answer protocol; the env must accept it."""
    env = MathAlgebraToolsEnv(conformal_quantile=0.5, max_tool_calls=0)
    assert env.max_tool_calls == 0


# ───────────────────────── tool schemas ─────────────────────────────


def test_tool_schemas_cover_four_primitives():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "sympy_simplify",
        "sympy_expand",
        "sympy_solve",
        "sympy_substitute",
    }


def test_tool_schemas_use_openai_function_calling_shape():
    """Each schema must be ``{type: function, function: {name, description, parameters}}``."""
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


# ───────────────────────── tool dispatch ────────────────────────────


def _toy_inst() -> Instance:
    return Instance(
        prompt="meta-test",
        gold_expr="x",
        seed=0,
        metadata={"alpha": 0.1, "simplify_timeout_s": 5.0, "template": "toy"},
    )


def test_dispatch_simplify_returns_canonical_form():
    inst = _toy_inst()
    out = dispatch_tool("sympy_simplify", {"expr_str": "(x+1)**2 - x**2 - 2*x"}, inst)
    assert "result" in out
    assert "error" not in out
    # Equivalent forms accepted; SymPy may render `1` or `1.0`.
    assert out["result"] in {"1", "1.0"}


def test_dispatch_expand_factored_to_polynomial():
    inst = _toy_inst()
    out = dispatch_tool("sympy_expand", {"expr_str": "(x+1)*(x-1)"}, inst)
    assert "result" in out
    assert out["result"].replace(" ", "") in {"x**2-1", "-1+x**2"}


def test_dispatch_solve_quadratic_returns_two_roots():
    inst = _toy_inst()
    out = dispatch_tool(
        "sympy_solve",
        {"equation_str": "x**2 - 4", "var_str": "x"},
        inst,
    )
    assert "solutions" in out
    sols = set(out["solutions"])
    assert sols == {"-2", "2"}


def test_dispatch_substitute_replaces_variable():
    inst = _toy_inst()
    out = dispatch_tool(
        "sympy_substitute",
        {"expr_str": "x**2 + x", "var_str": "x", "value_str": "3"},
        inst,
    )
    assert "result" in out
    assert out["result"] == "12"


def test_dispatch_unknown_tool_returns_error():
    inst = _toy_inst()
    out = dispatch_tool("nonexistent_tool", {}, inst)
    assert "error" in out
    assert "unknown tool" in out["error"]


def test_dispatch_invalid_json_arguments_returns_error():
    inst = _toy_inst()
    out = dispatch_tool("sympy_simplify", "{{not json", inst)
    assert "error" in out


def test_dispatch_unparseable_expr_returns_error():
    inst = _toy_inst()
    out = dispatch_tool("sympy_simplify", {"expr_str": "not sympy at all !"}, inst)
    assert "error" in out


def test_dispatch_accepts_json_string_arguments():
    """OpenAI tool-call protocol passes arguments as a JSON string."""
    inst = _toy_inst()
    out = dispatch_tool(
        "sympy_expand",
        json.dumps({"expr_str": "(x+1)*(x-1)"}),
        inst,
    )
    assert "result" in out


# ───────────────────────── adapter contract ─────────────────────────


def test_adapter_env_name_matches_registry_key():
    adapter = MathAlgebraToolsAdapter()
    assert adapter.env_name == NAME


def test_system_prompt_documents_four_primitives():
    for primitive in ("sympy_simplify", "sympy_expand", "sympy_solve", "sympy_substitute"):
        assert primitive in SYSTEM_PROMPT_TOOLS


def test_adapter_parse_response_round_trip():
    inst = generate_instance(seed=0)
    adapter = MathAlgebraToolsAdapter()
    payload = {"answer": "x", "confidence": 0.5}
    pred = adapter.parse_response(json.dumps(payload), inst)
    assert pred.answer_expr == "x"
    assert pred.confidence == pytest.approx(0.5)


def test_adapter_registered_globally():
    from verifiable_labs_envs.solvers import adapters  # noqa: F401  triggers registration
    from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

    assert NAME in _ADAPTERS


# ───────────────────────── verifier reuse ───────────────────────────


def test_score_delegates_to_single_turn_reward():
    """Tools env produces the same final-answer reward as the single-turn env."""
    inst = generate_instance(seed=11)
    pred = Prediction(
        answer_expr=inst.gold_expr,
        raw=json.dumps({"answer": inst.gold_expr, "confidence": 1.0}),
        confidence=1.0,
    )
    single = MathAlgebraEnv(conformal_quantile=0.5)
    tools = MathAlgebraToolsEnv(conformal_quantile=0.5)
    s_out = single.score(pred, inst)
    t_out = tools.score(pred, inst)
    assert s_out["reward"] == t_out["reward"]
