"""LLM adapter for the math-algebra environment.

The math-algebra env is single-turn: the solver reads a natural-language
algebra problem and returns a JSON envelope ``{"answer": "<sympy>", "confidence": <float>}``.
This adapter handles the prompt rendering and the (permissive) JSON
extraction so the env can be driven end-to-end by ``verifiable run --env
math-algebra --agent <provider:model>``.

The adapter is intentionally tolerant of common LLM output noise —
markdown fences, leading prose, numeric-vs-string confidence — so a
weak baseline still produces a parseable :class:`Prediction` rather
than wedging the rollout. Equivalence checking happens downstream in
``MathAlgebraEnv.score`` via SymPy.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.math_algebra import Instance, Prediction
from verifiable_labs_envs.solvers.adapters._common import extract_json_block
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter

SYSTEM_PROMPT = """You are an expert at symbolic algebra.

You will be given an algebra problem (expand, factor, or simplify an
expression) and must return your final answer as a SymPy-parseable
string, plus a self-reported confidence in [0, 1].

Output exactly one JSON object matching the schema given in the user
message. No prose, no markdown fences, no explanations.

Example output:
{"answer": "x**2 - 1", "confidence": 0.9}"""


def _build_user_prompt(instance: Instance) -> str:
    return (
        "PROBLEM:\n"
        + str(instance.prompt)
        + "\n\nOUTPUT SCHEMA:\n"
        + '{"answer": "<sympy-parseable string, e.g. x**2 - 1>",\n'
        + ' "confidence": <float in [0, 1]>}'
        + "\n\nRespond with the JSON object only."
    )


def _parse_response(text: str, instance: Instance) -> Prediction:
    """Parse the LLM response into a :class:`Prediction`.

    Permissive but non-silent: the JSON block must be extractable
    (raises :class:`LLMSolverError` if not). A missing or malformed
    ``answer`` field still yields a Prediction with empty answer (zero
    reward) rather than raising — matches the inverse-problem family's
    behaviour where a syntactically-bad solver output is captured as
    a zero-reward attempt rather than a hard failure.
    """
    del instance  # not needed — symbolic envs have no shape to match
    parsed = extract_json_block(text)

    answer_raw = parsed.get("answer", "")
    answer_expr = "" if answer_raw is None else str(answer_raw).strip()

    conf_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence != confidence or confidence in (float("inf"), float("-inf")):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return Prediction(
        answer_expr=answer_expr,
        raw=text,
        confidence=confidence,
    )


class MathAlgebraLLMAdapter(EnvAdapter):
    env_name: str = "math-algebra"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return _build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return _parse_response(text, instance)


__all__ = ["MathAlgebraLLMAdapter", "SYSTEM_PROMPT"]
