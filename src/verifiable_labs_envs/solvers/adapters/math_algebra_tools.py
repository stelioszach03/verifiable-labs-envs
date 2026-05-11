"""LLM adapter for the tool-use math-algebra environment.

Reuses :class:`MathAlgebraLLMAdapter` for prompt building + final-answer
parsing. The system prompt is rewritten to document the four SymPy
primitives and the OpenAI-style function-calling protocol.
"""
from __future__ import annotations

from typing import Any

# NOTE: ``TOOL_SCHEMAS`` is imported lazily inside ``get_tools_schema``
# below — pulling it at module top creates a circular import path
# (``envs.math_algebra_tools`` → ``solvers.llm_solver`` →
# ``solvers.__init__`` → ``solvers.adapters.__init__`` → here → back to
# ``envs.math_algebra_tools`` mid-load) that triggers
# ``ImportError: cannot import name 'TOOL_SCHEMAS' from partially
# initialized module`` in a fresh interpreter. Reproduced by the Prime
# Intellect Hub test_install_and_import CI check.
from verifiable_labs_envs.solvers.adapters.math_algebra import MathAlgebraLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "math-algebra-tools"

SYSTEM_PROMPT_TOOLS = """You are an expert at symbolic algebra.

You have access to 4 SymPy primitive tools. NONE of them returns a
final answer on its own — each performs one focused operation. You
compose them yourself to reach the final canonical form.

Available primitives:

- sympy_simplify(expr_str):
    Apply sympy.simplify to an expression string. Returns the simplified
    form. Useful for collapsing equivalent forms.

- sympy_expand(expr_str):
    Apply sympy.expand. Useful for products → polynomials, e.g.
    (x+1)*(x-1) → x**2 - 1.

- sympy_solve(equation_str, var_str):
    Solve a single-variable equation (treated as `equation_str = 0`)
    for the named variable. Returns a list of solution strings.

- sympy_substitute(expr_str, var_str, value_str):
    Substitute `var = value` in `expr_str`. Returns the substituted form.

Each primitive runs with a 5-second hard timeout. Adversarial inputs
return an `error` field instead of a result.

Recipe — for an expansion problem you might:
    expand((x+a)*(x+b))   →   composition of sympy_expand + verification
    via sympy_simplify on the difference between your candidate and the
    expanded form.

You may emit up to 30 tool calls per episode. After them (or earlier,
when you are confident), output a final JSON object:

    {"answer": "<sympy-parseable string>", "confidence": <float in [0, 1]>}

No prose, no markdown fences around the final answer."""


class MathAlgebraToolsAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_TOOLS

    def __init__(self) -> None:
        self._base = MathAlgebraLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)

    def get_tools_schema(self, instance: Any) -> list[dict[str, Any]] | None:
        """Forward the env's 4-primitive SymPy tool schema."""
        del instance  # the env exposes a static pool to every instance
        # Lazy import — see module-top NOTE on circular path.
        from verifiable_labs_envs.envs.math_algebra_tools import TOOL_SCHEMAS

        return list(TOOL_SCHEMAS)


__all__ = ["MathAlgebraToolsAdapter", "NAME", "SYSTEM_PROMPT_TOOLS"]
