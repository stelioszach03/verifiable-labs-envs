"""Environment 3b — multi-turn LoDoPaB-CT reconstruction.

Extends ``lodopab-ct-simplified`` with a 3-turn rollout:

* **Turn 1**: the LLM sees a coarse 32x32 FBP reconstruction of the sinogram
  and proposes a 32x32 refined image (the same task as the single-turn env).
* **Turn 2**: the server computes the sinogram residual ``r = y - R(x_hat)``
  of the LLM's upsampled answer, back-projects it to image space via FBP,
  downsamples to 32x32, and feeds it back as user content. The LLM proposes
  a corrected 32x32 image.
* **Turn 3**: same, final answer scored.

Reward is computed on the **final** turn's prediction after server-side
bicubic upsampling to the phantom shape. ``meta.turn_rewards`` and
``meta.turn_components`` expose the per-turn trajectory.

Works in both ``use_real_data=False`` (synthetic phantoms, CI default) and
``use_real_data=True`` (LoDoPaB-CT validation slices).
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.lodopab_ct import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.lodopab_ct import (
    Instance,
    LodopabCtEnv,
    Prediction,
    _cached_quantile,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "lodopab-ct-simplified-multiturn"
DEFAULT_MAX_TURNS: int = 3


class LodopabCtMultiturnEnv(LodopabCtEnv):
    """``LodopabCtEnv`` with a multi-turn rollout entry point. Supports both
    phantom and real-data modes via the inherited ``use_real_data`` flag."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        use_real_data: bool = False,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights, use_real_data=use_real_data)
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
        """Run up to ``max_turns`` turns. Semantics mirror the sparse-F multi-turn env.

        On parse failure after at least one successful turn: halt, score the
        last good prediction, return with ``n_turns`` reflecting the partial
        success. On first-turn failure: ``LLMSolverError`` propagates.
        """
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
    use_real_data: bool = False,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> LodopabCtMultiturnEnv:
    """Factory matching ``lodopab_ct.load_environment`` signature."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(float(_BASE_DEFAULTS["alpha"]), 3)
    else:
        q = _cached_quantile(float(_BASE_DEFAULTS["alpha"]), None)
    return LodopabCtMultiturnEnv(
        conformal_quantile=q,
        use_real_data=use_real_data,
        max_turns=max_turns,
    )


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "LodopabCtMultiturnEnv",
    "load_environment",
]
