"""Multi-turn phase retrieval — 3-turn rollout with magnitude-residual feedback.

Extends ``phase-retrieval`` with a 3-turn dialogue:

* Turn 1: LLM sees (n, k, sigma, mask, y_mag) and proposes (support_idx, support_amp_x1000).
* Turn 2: server computes the magnitude-domain residual
  ``r_mag = y - |F(x_hat)|`` on the LLM's answer and feeds it back. The LLM
  revises.
* Turn 3: same, final answer scored.

Reward on the final turn. ``meta.turn_rewards`` / ``meta.turn_components``
expose the per-turn trajectory for plot-ready data.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.phase_retrieval import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.phase_retrieval import (
    Instance,
    PhaseRetrievalEnv,
    Prediction,
    _cached_quantile,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "phase-retrieval-multiturn"
DEFAULT_MAX_TURNS: int = 3


class PhaseRetrievalMultiturnEnv(PhaseRetrievalEnv):
    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights)
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1; got {max_turns}")
        self.max_turns = int(max_turns)

    def run_rollout(
        self,
        solver: LLMSolver,
        instance: Instance,
        *,
        adapter: EnvAdapter | None = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        turns = int(max_turns or self.max_turns)

        history: list[dict[str, str]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        turn_rewards: list[float] = []
        turn_components: list[dict[str, float]] = []
        last_prediction: Prediction | None = None

        for turn_idx in range(turns):
            completion = solver.complete_turns(history)
            try:
                prediction = adapter.parse_response(completion.text, instance)
            except LLMSolverError:
                if last_prediction is None:
                    raise
                break

            scored = self.score(prediction, instance)
            turn_rewards.append(float(scored["reward"]))
            turn_components.append(dict(scored["components"]))
            last_prediction = prediction

            if turn_idx + 1 < turns:
                history.append({"role": "assistant", "content": completion.text})
                history.append({
                    "role": "user",
                    "content": adapter.build_followup_turn(history, prediction, instance),
                })

        assert last_prediction is not None
        final = self.score(last_prediction, instance)
        final["meta"] = {
            **final["meta"],
            "turn_rewards": turn_rewards,
            "turn_components": turn_components,
            "n_turns": len(turn_rewards),
            "max_turns": turns,
        }
        return final


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> PhaseRetrievalMultiturnEnv:
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(
            n_samples=30, alpha=float(_BASE_DEFAULTS["alpha"]), gs_iters=50,
        )
    else:
        q = _cached_quantile(
            n_samples=200, alpha=float(_BASE_DEFAULTS["alpha"]), gs_iters=200,
        )
    return PhaseRetrievalMultiturnEnv(conformal_quantile=q, max_turns=max_turns)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "PhaseRetrievalMultiturnEnv",
    "load_environment",
]
