"""Tests for the phase-retrieval env (sprint-giga Task 1)."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_envs import list_environments, load_environment
from verifiable_labs_envs.envs.phase_retrieval import (
    PhaseRetrievalEnv,
    _project_k_sparse,
    _sign_invariant_nmse,
    _support_f1,
    generate_instance,
    gerchberg_saxton_baseline,
    zero_baseline,
)
from verifiable_labs_envs.envs.phase_retrieval_multiturn import (
    PhaseRetrievalMultiturnEnv,
)
from verifiable_labs_envs.envs.phase_retrieval_multiturn import (
    load_environment as load_mt,
)
from verifiable_labs_envs.solvers.adapters.phase_retrieval import (
    PhaseRetrievalLLMAdapter,
)
from verifiable_labs_envs.solvers.adapters.phase_retrieval_multiturn import (
    PhaseRetrievalMultiturnAdapter,
)

ENV_NAME = "phase-retrieval"
MT_ENV_NAME = "phase-retrieval-multiturn"


# ---------- Registry ----------


def test_phase_retrieval_env_registered():
    assert ENV_NAME in list_environments()
    assert MT_ENV_NAME in list_environments()


def test_load_environment_returns_phase_env():
    env = load_environment(ENV_NAME, calibration_quantile=2.0)
    assert isinstance(env, PhaseRetrievalEnv)
    assert env.name == ENV_NAME


def test_load_environment_returns_mt_env():
    env = load_environment(MT_ENV_NAME, calibration_quantile=2.0)
    assert isinstance(env, PhaseRetrievalMultiturnEnv)
    assert env.name == MT_ENV_NAME


# ---------- Instance generation ----------


def test_generate_instance_produces_correct_shapes():
    inst = generate_instance(seed=0)
    assert inst.x_true.shape == (inst.n,)
    assert inst.y.shape == (inst.mask.size,)
    assert inst.support_true.shape == (inst.k,)
    assert np.all(inst.y >= 0.0)  # magnitudes are non-negative


def test_generate_instance_seed_determinism():
    a = generate_instance(seed=42)
    b = generate_instance(seed=42)
    assert np.array_equal(a.x_true, b.x_true)
    assert np.allclose(a.y, b.y)


def test_generate_instance_different_seeds_differ():
    a = generate_instance(seed=0)
    b = generate_instance(seed=1)
    assert not np.array_equal(a.x_true, b.x_true)


def test_x_true_is_exactly_k_sparse():
    inst = generate_instance(seed=0)
    nz = np.count_nonzero(inst.x_true)
    assert nz == inst.k


# ---------- Helpers ----------


def test_project_k_sparse_keeps_top_k_by_abs():
    x = np.array([0.1, -2.0, 0.5, 3.0, -0.2])
    projected, support = _project_k_sparse(x, k=2)
    assert set(support.tolist()) == {1, 3}
    assert projected[1] == -2.0
    assert projected[3] == 3.0
    assert projected[0] == 0.0
    assert projected[4] == 0.0


def test_sign_invariant_nmse_zero_when_x_hat_equals_x_true():
    x = np.array([1.0, -2.0, 0.0])
    assert _sign_invariant_nmse(x, x) == pytest.approx(0.0)


def test_sign_invariant_nmse_zero_when_x_hat_is_negated_x_true():
    x = np.array([1.0, -2.0, 0.0])
    assert _sign_invariant_nmse(x, -x) == pytest.approx(0.0)


def test_sign_invariant_nmse_picks_the_better_sign():
    x = np.array([1.0, 2.0, 3.0])
    x_hat = np.array([-1.0, -2.0, -3.0])  # exactly -x
    plus = np.sum((x - x_hat) ** 2) / np.sum(x ** 2)  # large
    minus = np.sum((x + x_hat) ** 2) / np.sum(x ** 2)  # zero
    assert _sign_invariant_nmse(x, x_hat) == pytest.approx(min(plus, minus))


def test_support_f1_perfect_recovery():
    support_true = np.array([0, 3, 7])
    x_hat = np.zeros(10)
    x_hat[support_true] = 1.0
    assert _support_f1(support_true, support_true, x_hat, 3) == pytest.approx(1.0)


# ---------- Baselines ----------


def test_zero_baseline_returns_zero_prediction():
    inst = generate_instance(seed=0)
    pred = zero_baseline(**inst.as_inputs())
    assert np.all(pred.x_hat == 0.0)
    assert pred.support_hat.size == 0


def test_gs_baseline_produces_k_sparse_prediction():
    inst = generate_instance(seed=0, n=16, k=3, m=12)
    pred = gerchberg_saxton_baseline(**inst.as_inputs(), gs_iters=100, n_restarts=5)
    assert pred.support_hat.size == inst.k
    assert np.count_nonzero(pred.x_hat) == inst.k


def test_gs_baseline_beats_zero_on_average():
    env = PhaseRetrievalEnv(conformal_quantile=2.0, hyperparams={"n": 16, "m": 12, "k": 3})
    gs_rewards = []
    zero_rewards = []
    for seed in range(5):
        inst = env.generate_instance(seed=seed)
        gs_rewards.append(env.score(
            gerchberg_saxton_baseline(**inst.as_inputs(), gs_iters=50, n_restarts=3), inst
        )["reward"])
        zero_rewards.append(env.score(zero_baseline(**inst.as_inputs()), inst)["reward"])
    # GS should on average be at least as good as zero.
    assert np.mean(gs_rewards) >= np.mean(zero_rewards)


# ---------- Env scoring ----------


def test_score_is_phase_invariant_under_sign_flip():
    env = PhaseRetrievalEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    # Build a "perfect" prediction
    from verifiable_labs_envs.envs.phase_retrieval import Prediction

    # And its negation: should score identically
    pred_plus = Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.full(inst.n, 0.1),
        support_hat=inst.support_true.copy(),
    )
    pred_minus = Prediction(
        x_hat=-inst.x_true.copy(),
        sigma_hat=np.full(inst.n, 0.1),
        support_hat=inst.support_true.copy(),
    )
    r_plus = env.score(pred_plus, inst)["reward"]
    r_minus = env.score(pred_minus, inst)["reward"]
    assert r_plus == pytest.approx(r_minus, abs=1e-9)


def test_perfect_prediction_scores_high():
    env = PhaseRetrievalEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    from verifiable_labs_envs.envs.phase_retrieval import Prediction

    pred = Prediction(
        x_hat=inst.x_true.copy(),
        sigma_hat=np.full(inst.n, 0.1),
        support_hat=inst.support_true.copy(),
    )
    out = env.score(pred, inst)
    assert out["reward"] > 0.9  # ~1 on nmse + support + conformal


def test_score_components_in_range():
    env = PhaseRetrievalEnv(conformal_quantile=2.0)
    inst = env.generate_instance(seed=0)
    pred = gerchberg_saxton_baseline(**inst.as_inputs(), gs_iters=50, n_restarts=3)
    out = env.score(pred, inst)
    for name, val in out["components"].items():
        assert 0.0 <= val <= 1.0, f"component {name}={val} out of [0, 1]"


# ---------- Adapter ----------


def test_adapter_builds_and_parses_truth():
    import json
    adapter = PhaseRetrievalLLMAdapter()
    inst = generate_instance(seed=0)
    prompt = adapter.build_user_prompt(inst)
    assert "y_mag_x1000" in prompt
    assert "support_idx" in prompt

    truth_json = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [int(round(v * 1000)) for v in inst.x_true[inst.support_true]],
    })
    pred = adapter.parse_response(truth_json, inst)
    assert pred.support_hat is not None
    assert pred.support_hat.size == inst.k


def test_mt_adapter_builds_followup_with_residual():
    import json
    adapter = PhaseRetrievalMultiturnAdapter()
    inst = generate_instance(seed=0)
    truth_json = json.dumps({
        "support_idx": [int(i) for i in inst.support_true],
        "support_amp_x1000": [int(round(v * 1000)) for v in inst.x_true[inst.support_true]],
    })
    pred = adapter.parse_response(truth_json, inst)
    followup = adapter.build_followup_turn([], pred, inst)
    assert "residual_mag_x1000" in followup
    assert "residual_l2_x1000" in followup


# ---------- Multi-turn env ----------


def test_mt_env_respects_max_turns_cap():
    env = PhaseRetrievalMultiturnEnv(conformal_quantile=2.0, max_turns=2)
    assert env.max_turns == 2


def test_mt_env_rejects_zero_max_turns():
    with pytest.raises(ValueError):
        PhaseRetrievalMultiturnEnv(conformal_quantile=2.0, max_turns=0)


def test_mt_load_environment_factory():
    env = load_mt(calibration_quantile=2.0, max_turns=3)
    assert env.max_turns == 3
