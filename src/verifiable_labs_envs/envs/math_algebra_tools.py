"""Environment — tool-use algebraic simplification (Phase 21.E).

The LLM solves the standard `math-algebra` problem but must compose
its answer using primitive SymPy tools rather than reasoning end-to-end.

Available tools (all timeout-bounded so adversarial inputs cannot wedge):

- ``sympy_simplify(expr_str)``                      — apply ``sympy.simplify``
- ``sympy_expand(expr_str)``                        — apply ``sympy.expand``
- ``sympy_solve(equation_str, var_str)``            — solve for ``var``
- ``sympy_substitute(expr_str, var_str, value_str)`` — substitute ``var = value``

No oracle. The model composes these into a final answer; reward is
computed on the final parsed ``answer`` exactly as in
:mod:`verifiable_labs_envs.envs.math_algebra` — tool calls do not
score directly, they are the *means* by which the model converges.

Procedural-regeneration contract is unchanged from `math-algebra` —
:class:`MathAlgebraToolsEnv` reuses the same problem generator,
verifier, and conformal calibration. Only the rollout protocol differs.
"""
from __future__ import annotations

import concurrent.futures
import json
from typing import Any

import sympy as sp

from verifiable_labs_envs.envs.math_algebra import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.math_algebra import (
    Instance,
    MathAlgebraEnv,
    Prediction,
    _cached_quantile,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "math-algebra-tools"
DEFAULT_MAX_TOOL_CALLS: int = 30
DEFAULT_TOOL_TIMEOUT_S: float = 5.0


# ──────────────────────────────────────────
# Tool schemas (OpenAI function-calling JSON)
# ──────────────────────────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sympy_simplify",
            "description": (
                "Apply sympy.simplify to a SymPy-parseable expression string. "
                "Returns the simplified form as a string, or an error if the "
                "input cannot be parsed or simplification times out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "SymPy-parseable expression, e.g. 'x**2 + 2*x + 1'.",
                    },
                },
                "required": ["expr_str"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sympy_expand",
            "description": (
                "Apply sympy.expand to a SymPy-parseable expression string. "
                "Useful for expanding products like (x+1)*(x-1) → x**2 - 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "SymPy-parseable expression to expand.",
                    },
                },
                "required": ["expr_str"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sympy_solve",
            "description": (
                "Solve a single-variable equation for the named variable. "
                "Returns a list of solution strings, or an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation_str": {
                        "type": "string",
                        "description": "Equation as a SymPy-parseable expression equal to 0, e.g. 'x**2 - 4'.",
                    },
                    "var_str": {
                        "type": "string",
                        "description": "Name of the variable to solve for, e.g. 'x'.",
                    },
                },
                "required": ["equation_str", "var_str"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sympy_substitute",
            "description": (
                "Substitute a value for a variable in a SymPy expression. "
                "Returns the substituted form as a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "SymPy-parseable expression.",
                    },
                    "var_str": {
                        "type": "string",
                        "description": "Variable name to substitute.",
                    },
                    "value_str": {
                        "type": "string",
                        "description": "SymPy-parseable replacement value.",
                    },
                },
                "required": ["expr_str", "var_str", "value_str"],
                "additionalProperties": False,
            },
        },
    },
]


# ──────────────────────────────────────────
# Tool implementations (timeout-wrapped)
# ──────────────────────────────────────────


def _run_with_timeout(
    fn: Any,
    *args: Any,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
) -> tuple[Any, str | None]:
    """Run ``fn(*args)`` in a daemon thread with a hard timeout.

    Returns ``(result, None)`` on success, ``(None, error_str)`` on
    timeout or any internal SymPy error. Cross-platform; same pattern
    as :func:`verifiable_labs_envs.envs.math_algebra.simplify_with_timeout`.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=timeout_s), None
        except concurrent.futures.TimeoutError:
            return None, f"timeout after {timeout_s}s"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


def _tool_simplify(args: dict[str, Any]) -> dict[str, Any]:
    expr_raw = args.get("expr_str", "")
    if not expr_raw:
        return {"error": "missing required argument expr_str"}
    try:
        expr = sp.sympify(expr_raw)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"error": f"could not parse expr_str: {exc}"}
    result, err = _run_with_timeout(sp.simplify, expr)
    if err:
        return {"error": err}
    return {"result": sp.sstr(result)}


def _tool_expand(args: dict[str, Any]) -> dict[str, Any]:
    expr_raw = args.get("expr_str", "")
    if not expr_raw:
        return {"error": "missing required argument expr_str"}
    try:
        expr = sp.sympify(expr_raw)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"error": f"could not parse expr_str: {exc}"}
    result, err = _run_with_timeout(sp.expand, expr)
    if err:
        return {"error": err}
    return {"result": sp.sstr(result)}


def _tool_solve(args: dict[str, Any]) -> dict[str, Any]:
    equation_raw = args.get("equation_str", "")
    var_raw = args.get("var_str", "")
    if not equation_raw or not var_raw:
        return {"error": "missing required arguments equation_str and/or var_str"}
    try:
        equation = sp.sympify(equation_raw)
        var = sp.Symbol(var_raw)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"error": f"could not parse arguments: {exc}"}
    result, err = _run_with_timeout(sp.solve, equation, var)
    if err:
        return {"error": err}
    if not isinstance(result, list):
        result = [result]
    return {"solutions": [sp.sstr(s) for s in result]}


def _tool_substitute(args: dict[str, Any]) -> dict[str, Any]:
    expr_raw = args.get("expr_str", "")
    var_raw = args.get("var_str", "")
    value_raw = args.get("value_str", "")
    if not expr_raw or not var_raw or not value_raw:
        return {"error": "missing required arguments expr_str, var_str, and/or value_str"}
    try:
        expr = sp.sympify(expr_raw)
        var = sp.Symbol(var_raw)
        value = sp.sympify(value_raw)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return {"error": f"could not parse arguments: {exc}"}

    def _subst() -> sp.Expr:
        return expr.subs(var, value)

    result, err = _run_with_timeout(_subst)
    if err:
        return {"error": err}
    return {"result": sp.sstr(result)}


_TOOL_DISPATCH = {
    "sympy_simplify": _tool_simplify,
    "sympy_expand": _tool_expand,
    "sympy_solve": _tool_solve,
    "sympy_substitute": _tool_substitute,
}


def dispatch_tool(name: str, arguments: str | dict, instance: Instance) -> dict[str, Any]:
    """Dispatch a tool call. ``arguments`` may be a JSON string (OpenAI format) or a dict."""
    del instance  # math envs don't need instance context inside tools
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"error": f"invalid JSON arguments: {exc}"}
    else:
        args = arguments or {}
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    return handler(args)


# ──────────────────────────────────────────
# Env
# ──────────────────────────────────────────


class MathAlgebraToolsEnv(MathAlgebraEnv):
    """:class:`MathAlgebraEnv` with a tool-use rollout entry point."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights)
        if max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0; got {max_tool_calls}")
        self.max_tool_calls = int(max_tool_calls)

    def run_rollout(
        self,
        solver: LLMSolver,
        instance: Instance,
        *,
        adapter: EnvAdapter | None = None,
        max_tool_calls: int | None = None,
    ) -> dict[str, Any]:
        """Run a tool-use rollout: alternating tool calls + final-answer turn.

        Returns the standard :meth:`score` dict with these extras in
        ``meta``:
            - ``tool_calls``: list[dict], each ``{"name": ..., "result": ...}``.
            - ``n_tool_calls``: int.
            - ``max_tool_calls``: int.
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        budget = int(max_tool_calls if max_tool_calls is not None else self.max_tool_calls)

        history: list[dict[str, str]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls: list[dict[str, Any]] = []
        last_prediction: Prediction | None = None

        for _ in range(budget + 1):  # budget tool calls + 1 final answer turn
            completion = solver.complete_turns(history, tools=TOOL_SCHEMAS)
            tool_call = getattr(completion, "tool_call", None)
            if tool_call is not None and len(tool_calls) < budget:
                result = dispatch_tool(tool_call.name, tool_call.arguments, instance)
                tool_calls.append({"name": tool_call.name, "result": result})
                history.append({"role": "assistant", "content": completion.text or ""})
                history.append({
                    "role": "tool",
                    "name": tool_call.name,
                    "content": json.dumps(result),
                })
                continue
            try:
                last_prediction = adapter.parse_response(completion.text, instance)
            except LLMSolverError:
                if last_prediction is None:
                    raise
            break

        assert last_prediction is not None
        final = self.score(last_prediction, instance)
        final["meta"] = {
            **final["meta"],
            "tool_calls": tool_calls,
            "n_tool_calls": len(tool_calls),
            "max_tool_calls": budget,
        }
        return final


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> MathAlgebraToolsEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    else:
        n = 30 if fast else 200
        q = _cached_quantile(n, float(_BASE_DEFAULTS["alpha"]))
    return MathAlgebraToolsEnv(conformal_quantile=q, max_tool_calls=max_tool_calls)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TOOL_CALLS",
    "TOOL_SCHEMAS",
    "MathAlgebraToolsEnv",
    "dispatch_tool",
    "load_environment",
]
