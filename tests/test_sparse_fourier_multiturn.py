"""Unit tests for the multi-turn sparse-Fourier environment."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verifiable_labs_envs import list_environments
from verifiable_labs_envs.envs import sparse_fourier as sf
from verifiable_labs_envs.envs.sparse_fourier_multiturn import (
    SparseFourierMultiturnEnv,
    load_environment,
)
from verifiable_labs_envs.solvers import FakeLLMSolver, LLMSolverError
from verifiable_labs_envs.solvers.adapters.sparse_fourier_multiturn import (
    SparseFourierMultiturnAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

ENV_NAME = "sparse-fourier-recovery-multiturn"


def _json_answer_from_support(support: np.ndarray, amps: np.ndarray) -> str:
    return json.dumps({
        "support_idx": [int(i) for i in support],
        "support_amp_x1000": [int(round(v * 1000)) for v in amps],
    })


def _truth_answer(instance: sf.Instance) -> str:
    return _json_answer_from_support(instance.support_true, instance.x_true[instance.support_true])


# ---------- Registry ----------


def test_multiturn_env_registered() -> None:
    assert ENV_NAME in list_environments()


def test_multiturn_adapter_registered() -> None:
    assert ENV_NAME in _ADAPTERS
    assert isinstance(_ADAPTERS[ENV_NAME], SparseFourierMultiturnAdapter)


# ---------- Instance + scoring delegation ----------


def test_generate_instance_delegates_to_base() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    assert inst.n == 256
    assert inst.k == 10
    assert inst.support_true.shape == (10,)


def test_score_delegates_to_base() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    x_hat = np.zeros(inst.n)
    sigma_hat = np.ones(inst.n)
    pred = sf.Prediction(x_hat=x_hat, sigma_hat=sigma_hat, support_hat=np.array([], dtype=np.int64))
    out = env.score(pred, inst)
    assert 0.0 <= out["reward"] <= 1.0
    assert "nmse" in out["components"]


# ---------- Rollout ----------


def test_run_rollout_three_turns_with_ground_truth_answers() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    answer = _truth_answer(inst)
    solver = FakeLLMSolver([answer, answer, answer])

    out = env.run_rollout(solver, inst)
    assert out["meta"]["n_turns"] == 3
    assert out["meta"]["max_turns"] == 3
    assert len(out["meta"]["turn_rewards"]) == 3
    assert all(r >= 0.9 for r in out["meta"]["turn_rewards"])


def test_run_rollout_builds_assistant_followup_pairs_in_history() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    answer = _truth_answer(inst)
    solver = FakeLLMSolver([answer, answer, answer])

    env.run_rollout(solver, inst)
    history_lists = solver.turn_calls
    assert [len(h) for h in history_lists] == [2, 4, 6]
    assert history_lists[1][-2]["role"] == "assistant"
    assert history_lists[1][-1]["role"] == "user"
    assert history_lists[1][-1]["content"].startswith("FEEDBACK:")


def test_run_rollout_max_turns_override() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0, max_turns=3)
    inst = env.generate_instance(seed=0)
    answer = _truth_answer(inst)
    solver = FakeLLMSolver([answer, answer])

    out = env.run_rollout(solver, inst, max_turns=2)
    assert out["meta"]["n_turns"] == 2
    assert out["meta"]["max_turns"] == 2
    assert len(solver.turn_calls) == 2


def test_run_rollout_halts_gracefully_on_parse_failure_after_turn_one() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    answer = _truth_answer(inst)
    solver = FakeLLMSolver([answer, "not json at all", "{}"])

    out = env.run_rollout(solver, inst)
    assert out["meta"]["n_turns"] == 1
    assert len(out["meta"]["turn_rewards"]) == 1
    assert out["reward"] == pytest.approx(out["meta"]["turn_rewards"][0])


def test_run_rollout_first_turn_failure_propagates() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver(["utter nonsense"])

    with pytest.raises(LLMSolverError):
        env.run_rollout(solver, inst)


def test_env_rejects_zero_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        SparseFourierMultiturnEnv(conformal_quantile=2.0, max_turns=0)


# ---------- Adapter followup content ----------


def test_build_followup_turn_emits_residual_payload() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    pred = sf.Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.ones(inst.n),
        support_hat=inst.support_true.copy(),
    )
    adapter = SparseFourierMultiturnAdapter()
    out = adapter.build_followup_turn(history=[], last_prediction=pred, instance=inst)
    assert out.startswith("FEEDBACK:")
    assert "residual_re_x1000" in out
    assert "residual_im_x1000" in out


def test_build_followup_turn_contains_mask_length_entries() -> None:
    env = SparseFourierMultiturnEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0, n=64, m=16, k=4)
    pred = sf.Prediction(
        x_hat=np.zeros(inst.n),
        sigma_hat=np.ones(inst.n),
        support_hat=np.array([], dtype=np.int64),
    )
    adapter = SparseFourierMultiturnAdapter()
    out = adapter.build_followup_turn(history=[], last_prediction=pred, instance=inst)
    start = out.index("{")
    end = out.rindex("}")
    payload = json.loads(out[start : end + 1])
    assert len(payload["residual_re_x1000"]) == inst.mask.size
    assert len(payload["residual_im_x1000"]) == inst.mask.size


# ---------- Factory ----------


def test_load_environment_returns_multiturn_env() -> None:
    env = load_environment(calibration_quantile=2.0, max_turns=2)
    assert isinstance(env, SparseFourierMultiturnEnv)
    assert env.max_turns == 2
    assert env.conformal_quantile == pytest.approx(2.0)


def test_load_environment_via_top_level_registry() -> None:
    from verifiable_labs_envs import load_environment as top_load

    env = top_load(ENV_NAME)
    assert isinstance(env, SparseFourierMultiturnEnv)
