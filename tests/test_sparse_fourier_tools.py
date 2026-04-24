"""Unit tests for the tool-use sparse-Fourier environment."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verifiable_labs_envs import list_environments
from verifiable_labs_envs.envs import sparse_fourier as sf
from verifiable_labs_envs.envs.sparse_fourier_tools import (
    TOOL_SCHEMAS,
    SparseFourierToolsEnv,
    dispatch_tool,
    load_environment,
)
from verifiable_labs_envs.solvers import FakeLLMSolver, LLMSolverError
from verifiable_labs_envs.solvers.adapters.sparse_fourier_tools import (
    SparseFourierToolsAdapter,
)
from verifiable_labs_envs.solvers.llm_solver import _ADAPTERS

ENV_NAME = "sparse-fourier-recovery-tools"


def _final_answer(instance: sf.Instance) -> str:
    return json.dumps({
        "support_idx": [int(i) for i in instance.support_true],
        "support_amp_x1000": [int(round(v * 1000)) for v in instance.x_true[instance.support_true]],
    })


def _tool_call_dict(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# ---------- Registry ----------


def test_tools_env_registered() -> None:
    assert ENV_NAME in list_environments()


def test_tools_adapter_registered() -> None:
    assert ENV_NAME in _ADAPTERS
    assert isinstance(_ADAPTERS[ENV_NAME], SparseFourierToolsAdapter)


def test_tool_schemas_are_well_formed() -> None:
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"fft_tool", "ifft_tool", "ista_tool", "check_residual_tool"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


# ---------- Tool executors ----------


def test_fft_tool_round_trips_truth() -> None:
    inst = sf.generate_instance(seed=0)
    truth_support = [int(i) for i in inst.support_true]
    truth_amp_x1000 = [int(round(v * 1000)) for v in inst.x_true[inst.support_true]]
    result = dispatch_tool("fft_tool",
                           {"support_idx": truth_support, "support_amp_x1000": truth_amp_x1000},
                           inst)
    assert "y_hat_re_x1000" in result
    assert "y_hat_im_x1000" in result
    assert len(result["y_hat_re_x1000"]) == inst.mask.size


def test_ifft_tool_rejects_wrong_length() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("ifft_tool",
                           {"real_x1000_at_mask": [1], "imag_x1000_at_mask": [2]},
                           inst)
    assert "error" in result
    assert "expected" in result["error"]


def test_ista_tool_returns_support_and_amp() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("ista_tool", {}, inst)
    assert "support_idx" in result
    assert "support_amp_x1000" in result
    assert len(result["support_idx"]) == inst.k


def test_check_residual_of_truth_is_small() -> None:
    inst = sf.generate_instance(seed=0)
    truth_support = [int(i) for i in inst.support_true]
    truth_amp_x1000 = [int(round(v * 1000)) for v in inst.x_true[inst.support_true]]
    result = dispatch_tool("check_residual_tool",
                           {"support_idx": truth_support, "support_amp_x1000": truth_amp_x1000},
                           inst)
    # Residual should be on the order of sigma*sqrt(m) * 1000 ~ small
    assert result["residual_l2_x1000"] < 1000


def test_dispatch_tool_unknown_name_returns_error() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("not_a_real_tool", {}, inst)
    assert "error" in result
    assert "unknown" in result["error"]


def test_dispatch_tool_accepts_json_string_arguments() -> None:
    inst = sf.generate_instance(seed=0)
    args_str = json.dumps({"support_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           "support_amp_x1000": [0] * 10})
    result = dispatch_tool("check_residual_tool", args_str, inst)
    assert "residual_l2_x1000" in result


# ---------- Rollout ----------


def test_run_rollout_zero_tool_calls_direct_answer() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=0)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver(_final_answer(inst))

    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 0
    assert out["meta"]["tool_sequence"] == []
    assert out["components"]["nmse"] > 0.9  # fed the truth


def test_run_rollout_with_one_tool_call_then_answer() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("ista_tool", {})]},  # first response: call ista
        _final_answer(inst),                                  # second: final answer
    ])

    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 1
    assert out["meta"]["tool_sequence"] == ["ista_tool"]
    assert out["components"]["nmse"] > 0.9


def test_run_rollout_respects_max_tool_calls_cap() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=2)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("ista_tool", {}, call_id="c1")]},
        {"tool_calls": [_tool_call_dict("ista_tool", {}, call_id="c2")]},
        {"tool_calls": [_tool_call_dict("ista_tool", {}, call_id="c3")]},  # over cap
        _final_answer(inst),
    ])
    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 2
    assert out["meta"]["max_tool_calls"] == 2


def test_run_rollout_records_tool_sequence_in_order() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("ifft_tool",
            {"real_x1000_at_mask": [0] * inst.mask.size,
             "imag_x1000_at_mask": [0] * inst.mask.size})]},
        {"tool_calls": [_tool_call_dict("check_residual_tool",
            {"support_idx": [0] * inst.k, "support_amp_x1000": [0] * inst.k})]},
        {"tool_calls": [_tool_call_dict("ista_tool", {})]},
        _final_answer(inst),
    ])
    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_sequence"] == ["ifft_tool", "check_residual_tool", "ista_tool"]


def test_run_rollout_propagates_parse_failure_on_final() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("ista_tool", {})]},
        "not JSON at all",  # final answer is broken
    ])
    with pytest.raises(LLMSolverError):
        env.run_rollout_with_tools(solver, inst)


# ---------- Factory ----------


def test_load_environment_returns_tools_env() -> None:
    env = load_environment(calibration_quantile=2.0, max_tool_calls=3)
    assert isinstance(env, SparseFourierToolsEnv)
    assert env.max_tool_calls == 3


def test_load_environment_via_top_level_registry() -> None:
    from verifiable_labs_envs import load_environment as top_load

    env = top_load(ENV_NAME, calibration_quantile=2.0)
    assert isinstance(env, SparseFourierToolsEnv)


def test_env_rejects_negative_max_tool_calls() -> None:
    with pytest.raises(ValueError, match="max_tool_calls"):
        SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=-1)


# ---------- Tool payload shape ----------


def test_fft_round_trip_through_ifft_is_sane() -> None:
    """For truth sparse x, fft_tool gives y_hat; ifft_tool on y_hat gives a signal
    whose L2 norm is nonzero."""
    inst = sf.generate_instance(seed=0)
    truth_support = [int(i) for i in inst.support_true]
    truth_amp_x1000 = [int(round(v * 1000)) for v in inst.x_true[inst.support_true]]
    fft_result = dispatch_tool("fft_tool",
                               {"support_idx": truth_support, "support_amp_x1000": truth_amp_x1000},
                               inst)
    ifft_result = dispatch_tool("ifft_tool",
                                {"real_x1000_at_mask": fft_result["y_hat_re_x1000"],
                                 "imag_x1000_at_mask": fft_result["y_hat_im_x1000"]},
                                inst)
    assert "signal_x1000" in ifft_result
    assert len(ifft_result["signal_x1000"]) == inst.n
    signal = np.array(ifft_result["signal_x1000"]) / 1000.0
    assert float(np.linalg.norm(signal)) > 0
