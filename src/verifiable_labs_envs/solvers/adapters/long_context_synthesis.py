"""LLM adapter for the long-context-synthesis environment (Phase 27).

The synthesis env is a 3-turn rollout. Turn 1 uses the same user
prompt as a single-turn variant; turns 2 and 3 receive the F1-bucketed
feedback string. The gold answer is NEVER serialised into the
feedback (R10 carry-over).
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.long_context_synthesis import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_response,
    render_synthesis_feedback,
    score_components,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter


class LongContextSynthesisAdapter(EnvAdapter):
    env_name: str = "long-context-synthesis"
    system_prompt: str = SYSTEM_PROMPT

    def build_user_prompt(self, instance: Any) -> str:
        return build_user_prompt(instance)

    def parse_response(self, text: str, instance: Any) -> Any:
        return parse_response(text, instance)

    def build_followup_turn(
        self,
        history: list[dict[str, str]],  # noqa: ARG002
        last_prediction: Any,
        instance: Any,
    ) -> str:
        if last_prediction is None:
            return render_synthesis_feedback(
                f1_score=0.0, needle_doc_ids=instance.needle_doc_ids,
            )
        components = score_components(last_prediction, instance)
        return render_synthesis_feedback(
            f1_score=float(components["correctness"]),
            needle_doc_ids=instance.needle_doc_ids,
        )


__all__ = ["LongContextSynthesisAdapter"]
