"""LLM adapter for the multi-turn math-algebra environment.

Reuses :class:`MathAlgebraLLMAdapter` for prompt building + parsing;
adds :meth:`build_followup_turn` that emits actionable verifier
feedback without revealing the gold expression.

The feedback ladder mirrors the reward kernel's short-circuit chain:

- ``format_valid == 0``  →  "your output was not valid JSON"
- ``parse_valid == 0``   →  "your answer was not a valid SymPy expression"
- ``correct == 0``       →  "your answer was not equivalent to the target"
- ``correct == 1``       →  "your previous answer was correct; you may keep it"
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.math_algebra import (
    Instance,
    Prediction,
    score_components,
)
from verifiable_labs_envs.solvers.adapters.math_algebra import MathAlgebraLLMAdapter
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

NAME = "math-algebra-multiturn"

SYSTEM_PROMPT_MT = """You are an expert at symbolic algebra.

You have up to 3 turns. On turn 1 you see the full problem and propose
an answer + a self-reported confidence.

On turns 2 and 3 you see FEEDBACK on your previous answer (whether it
was correct, and if not, at which validation step it failed). Use the
feedback to propose a corrected answer with the same JSON schema. The
gold expression itself is never revealed.

Always output exactly one JSON object:
{"answer": "<sympy-parseable string>", "confidence": <float in [0, 1]>}

No prose, no markdown fences, no explanations."""


def _build_feedback(prediction: Prediction, instance: Instance) -> str:
    """Render verifier feedback without leaking the gold expression."""
    components = score_components(prediction, instance)
    if components["format_valid"] == 0.0:
        diagnosis = (
            "Your output was not valid JSON. Return exactly one JSON object "
            "matching the schema: "
            '{"answer": "<sympy-parseable>", "confidence": <float>}.'
        )
    elif components["parse_valid"] == 0.0:
        diagnosis = (
            "Your `answer` field was not a valid SymPy expression. "
            "Use SymPy syntax: e.g. `x**2 - 1` (not `x^2 - 1`), `*` for "
            "multiplication, parentheses where needed."
        )
    elif components["correct"] == 0.0:
        diagnosis = (
            "Your answer parses but is not equivalent to the target. "
            "Re-examine the problem and try a different form."
        )
    else:
        diagnosis = (
            "Your previous answer was correct. You may submit the same "
            "answer (or an equivalent form) for the final turn."
        )
    return "FEEDBACK on your previous turn:\n" + diagnosis


class MathAlgebraMultiturnAdapter(EnvAdapter):
    env_name: str = NAME
    system_prompt: str = SYSTEM_PROMPT_MT

    def __init__(self) -> None:
        self._base = MathAlgebraLLMAdapter()

    def build_user_prompt(self, instance: Any) -> str:
        return self._base.build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return self._base.parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],  # noqa: ARG002
        last_prediction: Any,
        instance: Any,
    ) -> str:
        return _build_feedback(last_prediction, instance)


__all__ = ["MathAlgebraMultiturnAdapter", "SYSTEM_PROMPT_MT"]
