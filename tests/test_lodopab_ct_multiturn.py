"""Unit tests for the multi-turn lodopab-ct-simplified environment."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verifiable_labs_envs import list_environments
from verifiable_labs_envs.envs import lodopab_ct as ct
from verifiable_labs_envs.envs.lodopab_ct_multiturn import (
    LodopabCtMultiturnEnv,
    load_environment,
)
from verifiable_labs_envs.solvers import FakeLLMSolver, LLMSolverError
from verifiable_labs_envs.solvers.adapters.lodopab_ct import COARSE_SIZE
from verifiable_labs_envs.solvers.adapters.lodopab_ct_multiturn import (
    LodopabCtMultiturnAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

ENV_NAME = "lodopab-ct-simplified-multiturn"


def _grid_answer(grid_int: list[list[int]]) -> str:
    return json.dumps({"image": grid_int})


def _mid_grey_grid() -> list[list[int]]:
    return [[128] * COARSE_SIZE for _ in range(COARSE_SIZE)]


# ---------- Registry ----------


def test_multiturn_ct_env_registered() -> None:
    assert ENV_NAME in list_environments()


def test_multiturn_ct_adapter_registered() -> None:
    assert ENV_NAME in _ADAPTERS
    assert isinstance(_ADAPTERS[ENV_NAME], LodopabCtMultiturnAdapter)


# ---------- Instance + scoring delegation ----------


def test_generate_instance_delegates_to_base() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    assert inst.x_true.shape == inst.shape
    assert inst.y.shape == (inst.shape[0], inst.n_angles)


def test_score_delegates_to_base() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    pred = ct.fbp_baseline(**inst.as_inputs())
    out = env.score(pred, inst)
    assert 0.0 <= out["reward"] <= 1.0


# ---------- Rollout ----------


def test_run_rollout_three_turns_with_grey_answers() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    ans = _grid_answer(_mid_grey_grid())
    solver = FakeLLMSolver([ans, ans, ans])

    out = env.run_rollout(solver, inst)
    assert out["meta"]["n_turns"] == 3
    assert len(out["meta"]["turn_rewards"]) == 3
    assert out["meta"]["max_turns"] == 3
    # Grey prediction produces some nonzero reward but low psnr
    assert 0.0 <= out["reward"] <= 1.0


def test_run_rollout_builds_assistant_followup_pairs() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    ans = _grid_answer(_mid_grey_grid())
    solver = FakeLLMSolver([ans, ans, ans])

    env.run_rollout(solver, inst)
    history_lists = solver.turn_calls
    assert [len(h) for h in history_lists] == [2, 4, 6]
    assert history_lists[1][-2]["role"] == "assistant"
    assert history_lists[1][-1]["role"] == "user"
    assert history_lists[1][-1]["content"].startswith("FEEDBACK:")


def test_run_rollout_max_turns_override() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241, max_turns=3)
    inst = env.generate_instance(seed=0)
    ans = _grid_answer(_mid_grey_grid())
    solver = FakeLLMSolver([ans, ans])

    out = env.run_rollout(solver, inst, max_turns=2)
    assert out["meta"]["n_turns"] == 2
    assert out["meta"]["max_turns"] == 2
    assert len(solver.turn_calls) == 2


def test_run_rollout_halts_gracefully_on_parse_failure_after_turn_one() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    ans = _grid_answer(_mid_grey_grid())
    solver = FakeLLMSolver([ans, "not json", "{}"])

    out = env.run_rollout(solver, inst)
    assert out["meta"]["n_turns"] == 1
    assert len(out["meta"]["turn_rewards"]) == 1
    assert out["reward"] == pytest.approx(out["meta"]["turn_rewards"][0])


def test_run_rollout_first_turn_failure_propagates() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver(["this is not a JSON grid"])

    with pytest.raises(LLMSolverError):
        env.run_rollout(solver, inst)


def test_env_rejects_zero_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        LodopabCtMultiturnEnv(conformal_quantile=0.241, max_turns=0)


# ---------- Adapter followup content ----------


def test_build_followup_turn_emits_residual_grid() -> None:
    env = LodopabCtMultiturnEnv(conformal_quantile=0.241)
    inst = env.generate_instance(seed=0)
    # Trivial mid-grey prediction so residual is non-zero
    pred = ct.Prediction(
        x_hat=np.full(inst.shape, 0.5),
        sigma_hat=np.full(inst.shape, 0.1),
    )
    adapter = LodopabCtMultiturnAdapter()
    out = adapter.build_followup_turn(history=[], last_prediction=pred, instance=inst)
    assert out.startswith("FEEDBACK:")
    assert "residual_32x32_int8" in out
    assert "scale_abs_max" in out

    start = out.index("{")
    end = out.rindex("}")
    payload = json.loads(out[start : end + 1])
    grid = payload["residual_32x32_int8"]
    assert len(grid) == COARSE_SIZE
    assert all(len(row) == COARSE_SIZE for row in grid)
    assert all(-128 <= v <= 127 for row in grid for v in row)
    assert payload["scale_abs_max"] >= 0


# ---------- Factory ----------


def test_load_environment_returns_multiturn_env() -> None:
    env = load_environment(calibration_quantile=0.241, max_turns=2)
    assert isinstance(env, LodopabCtMultiturnEnv)
    assert env.max_turns == 2


def test_load_environment_via_top_level_registry() -> None:
    from verifiable_labs_envs import load_environment as top_load

    env = top_load(ENV_NAME, calibration_quantile=0.241)
    assert isinstance(env, LodopabCtMultiturnEnv)
