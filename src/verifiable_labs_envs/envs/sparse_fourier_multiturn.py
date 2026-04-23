"""Environment 1b — multi-turn sparse-Fourier recovery.

Extends ``sparse-fourier-recovery`` with a 3-turn rollout:

* **Turn 1**: the LLM sees the full problem ``(n, k, sigma, mask, y)`` and
  proposes ``(support_idx, support_amp_x1000)``.
* **Turn 2**: the server computes the Fourier-domain residual
  ``r = y - A(x_hat)`` on the LLM's first answer and feeds ``r`` back as the
  user message; the LLM proposes a correction.
* **Turn 3**: same, final answer.

Reward is computed on the **final** turn's prediction. ``meta.turn_rewards``
and ``meta.turn_components`` expose the per-turn trajectory so we can plot
"does the model improve per turn" in the v2 benchmark.

Ground-truth generation, forward operator, scoring, conformal calibration,
and the ``ista_baseline`` all delegate to the single-turn env —
``sparse_fourier_multiturn`` is a pure conversation wrapper around it.
"""
from __future__ import annotations

from typing import Any

from verifiable_labs_envs.envs.sparse_fourier import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.sparse_fourier import (
    Instance,
    Prediction,
    SparseFourierEnv,
    _cached_quantile,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "sparse-fourier-recovery-multiturn"
DEFAULT_MAX_TURNS: int = 3


class SparseFourierMultiturnEnv(SparseFourierEnv):
    """``SparseFourierEnv`` with a multi-turn rollout entry point."""

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
        """Run up to ``max_turns`` turns of ``solver`` on ``instance``.

        Returns the single-turn ``score(..)`` dict plus these extras in ``meta``:
            - ``turn_rewards``: list[float], the final reward after each turn.
            - ``turn_components``: list[dict], per-turn component breakdown.
            - ``n_turns``: int, the number of turns actually taken.
            - ``max_turns``: int, the cap for this rollout.

        If a turn returns un-parseable output after at least one successful
        turn, the rollout halts and the last good prediction is scored.
        If turn 1 is unparseable, ``LLMSolverError`` propagates to the caller —
        we don't fabricate partial credit for a solver that can't follow the
        documented schema even once.
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
    max_turns: int = DEFAULT_MAX_TURNS,
) -> SparseFourierMultiturnEnv:
    """Factory matching the single-turn env. Calibration is reused."""
    if calibration_quantile is not None:
        q = float(calibration_quantile)
    elif fast:
        q = _cached_quantile(
            n_samples=30,
            alpha=float(_BASE_DEFAULTS["alpha"]),
            n_bootstrap=5,
            n_iters=80,
        )
    else:
        q = _cached_quantile(
            n_samples=500,
            alpha=float(_BASE_DEFAULTS["alpha"]),
            n_bootstrap=20,
            n_iters=200,
        )
    return SparseFourierMultiturnEnv(conformal_quantile=q, max_turns=max_turns)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TURNS",
    "SparseFourierMultiturnEnv",
    "load_environment",
]
