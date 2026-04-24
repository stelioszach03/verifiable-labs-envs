"""Unit tests for the primitive-composition tool-use sparse-Fourier env."""
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

PRIMITIVE_NAMES = {
    "fft_tool",
    "ifft_tool",
    "threshold_tool",
    "compute_residual_tool",
    "sparsity_norm_tool",
}


def _final_answer(instance: sf.Instance) -> str:
    return json.dumps({
        "support_idx": [int(i) for i in instance.support_true],
        "support_amp_x1000": [int(round(v * 1000)) for v in instance.x_true[instance.support_true]],
    })


def _truth_dense_x1000(inst: sf.Instance) -> list[int]:
    return [int(round(float(v) * 1000)) for v in inst.x_true.tolist()]


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
    assert names == PRIMITIVE_NAMES
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "description" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_no_oracle_tool_in_schemas() -> None:
    """Guard against re-introducing a solver oracle tool."""
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "ista_tool" not in names
    assert "omp_tool" not in names
    assert "solve_tool" not in names


# ---------- Tool executors ----------


def test_fft_tool_accepts_dense_and_returns_m_coefficients() -> None:
    inst = sf.generate_instance(seed=0)
    signal = _truth_dense_x1000(inst)
    result = dispatch_tool("fft_tool", {"signal_x1000": signal}, inst)
    assert "y_hat_re_x1000" in result
    assert "y_hat_im_x1000" in result
    assert len(result["y_hat_re_x1000"]) == inst.mask.size
    assert len(result["y_hat_im_x1000"]) == inst.mask.size


def test_fft_tool_rejects_wrong_length() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("fft_tool", {"signal_x1000": [1, 2, 3]}, inst)
    assert "error" in result
    assert f"length {inst.n}" in result["error"]


def test_ifft_tool_rejects_wrong_spectrum_length() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool(
        "ifft_tool", {"spectrum_re_x1000": [1], "spectrum_im_x1000": [2]}, inst
    )
    assert "error" in result


def test_threshold_tool_soft_zeros_small_entries() -> None:
    inst = sf.generate_instance(seed=0)
    # signal with a few small and a few large entries
    signal = [50] * inst.n  # 0.050 everywhere
    signal[0] = 800  # 0.8
    signal[5] = -600  # -0.6
    result = dispatch_tool(
        "threshold_tool", {"signal_x1000": signal, "tau_x1000": 100}, inst
    )
    out = np.array(result["signal_x1000"]) / 1000.0
    # Entries at 0.050 get threshold 0.1 applied → zero
    assert float(np.abs(out).max()) <= 0.701  # 0.8 - 0.1 = 0.7, allow fp
    # The two big entries should survive with their sign
    assert out[0] > 0
    assert out[5] < 0
    # All others zero
    small_entries = np.delete(out, [0, 5])
    assert np.all(small_entries == 0.0)
    # nonzero_count reports 2
    assert result["nonzero_count"] == 2


def test_threshold_tool_rejects_negative_tau() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool(
        "threshold_tool",
        {"signal_x1000": [0] * inst.n, "tau_x1000": -5},
        inst,
    )
    assert "error" in result
    assert ">= 0" in result["error"]


def test_threshold_tool_rejects_missing_tau() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("threshold_tool", {"signal_x1000": [0] * inst.n}, inst)
    assert "error" in result
    assert "tau_x1000" in result["error"]


def test_compute_residual_of_truth_is_small() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool(
        "compute_residual_tool", {"signal_x1000": _truth_dense_x1000(inst)}, inst
    )
    # Residual should be on the order of sigma*sqrt(m) * 1000 ~ small
    assert result["residual_l2_x1000"] < 1000
    assert len(result["residual_re_x1000"]) == inst.mask.size
    assert len(result["residual_im_x1000"]) == inst.mask.size


def test_sparsity_norm_of_truth_has_k_nonzeros() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool(
        "sparsity_norm_tool", {"signal_x1000": _truth_dense_x1000(inst)}, inst
    )
    assert result["nonzero_count"] == inst.k
    assert result["l1_x1000"] > 0
    assert result["l2_x1000"] > 0


def test_sparsity_norm_of_zero_signal() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool(
        "sparsity_norm_tool", {"signal_x1000": [0] * inst.n}, inst
    )
    assert result["nonzero_count"] == 0
    assert result["l1_x1000"] == 0
    assert result["l2_x1000"] == 0


def test_dispatch_tool_unknown_name_returns_error() -> None:
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("not_a_real_tool", {}, inst)
    assert "error" in result
    assert "unknown" in result["error"]


def test_dispatch_tool_rejects_removed_ista_tool() -> None:
    """v0.1's ista_tool oracle is gone in v0.3 — dispatch must reject it."""
    inst = sf.generate_instance(seed=0)
    result = dispatch_tool("ista_tool", {}, inst)
    assert "error" in result
    assert "unknown" in result["error"]


def test_dispatch_tool_accepts_json_string_arguments() -> None:
    inst = sf.generate_instance(seed=0)
    args_str = json.dumps({"signal_x1000": [0] * inst.n})
    result = dispatch_tool("compute_residual_tool", args_str, inst)
    assert "residual_l2_x1000" in result


# ---------- Single-tool-call regression ----------


def test_no_single_tool_call_leaks_the_answer() -> None:
    """v0.1's ista_tool returned the OMP reconstruction; the model copied it
    verbatim and got reward ~0.86 from a single tool call. v0.3 must NOT
    have that property: a single primitive call followed by a fixed final
    answer should score identically to the same fixed answer with 0 tool
    calls — proving the primitive did not transmit the target to the model.

    Using the empty-answer baseline (amps all zero) as the fixed answer:
    its reward is ~0.35 (floor from conformal coverage + nmse); a primitive
    that leaked the answer would push reward well above that. Using 0.5 as
    the hard ceiling (floor ~0.35, generous slack, still catches the v0.1
    oracle at 0.86)."""
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=1)
    inst = env.generate_instance(seed=0)
    zeros_dense = [0] * inst.n
    empty_answer = json.dumps({
        "support_idx": sorted(range(inst.k)),
        "support_amp_x1000": [0] * inst.k,
    })

    # Baseline: empty answer with 0 tool calls.
    baseline_env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=0)
    baseline_out = baseline_env.run_rollout_with_tools(
        FakeLLMSolver(empty_answer), inst
    )
    baseline_reward = baseline_out["reward"]

    for tool_name, tool_args in [
        ("fft_tool", {"signal_x1000": zeros_dense}),
        ("ifft_tool",
         {"spectrum_re_x1000": [0] * inst.mask.size,
          "spectrum_im_x1000": [0] * inst.mask.size}),
        ("threshold_tool", {"signal_x1000": zeros_dense, "tau_x1000": 100}),
        ("compute_residual_tool", {"signal_x1000": zeros_dense}),
        ("sparsity_norm_tool", {"signal_x1000": zeros_dense}),
    ]:
        solver = FakeLLMSolver([
            {"tool_calls": [_tool_call_dict(tool_name, tool_args)]},
            empty_answer,
        ])
        out = env.run_rollout_with_tools(solver, inst)
        # Reward must match baseline exactly (same final answer → same score).
        assert abs(out["reward"] - baseline_reward) < 1e-9, (
            f"primitive {tool_name} changed reward despite fixed answer: "
            f"baseline={baseline_reward:.4f}, with-tool={out['reward']:.4f}"
        )
        # And must be under a generous absolute cap that catches v0.1's oracle.
        assert out["reward"] <= 0.5, (
            f"primitive {tool_name} single-call leaked a shortcut: "
            f"reward={out['reward']:.3f} > 0.5 — investigate before declaring v0.3"
        )


# ---------- Rollout harness ----------


def test_run_rollout_zero_tool_calls_direct_answer() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=0)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver(_final_answer(inst))

    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 0
    assert out["meta"]["tool_sequence"] == []
    assert out["components"]["nmse"] > 0.9  # fed the truth


def test_run_rollout_with_primitive_sequence_then_answer() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict(
            "ifft_tool",
            {"spectrum_re_x1000": [0] * inst.mask.size,
             "spectrum_im_x1000": [0] * inst.mask.size},
        )]},
        {"tool_calls": [_tool_call_dict(
            "threshold_tool",
            {"signal_x1000": [0] * inst.n, "tau_x1000": 100},
        )]},
        _final_answer(inst),
    ])

    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 2
    assert out["meta"]["tool_sequence"] == ["ifft_tool", "threshold_tool"]
    assert out["components"]["nmse"] > 0.9


def test_run_rollout_respects_max_tool_calls_cap() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=2)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("sparsity_norm_tool",
                                        {"signal_x1000": [0] * inst.n}, call_id="c1")]},
        {"tool_calls": [_tool_call_dict("sparsity_norm_tool",
                                        {"signal_x1000": [0] * inst.n}, call_id="c2")]},
        {"tool_calls": [_tool_call_dict("sparsity_norm_tool",
                                        {"signal_x1000": [0] * inst.n}, call_id="c3")]},
        _final_answer(inst),
    ])
    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_calls"] == 2
    assert out["meta"]["max_tool_calls"] == 2


def test_run_rollout_records_tool_sequence_in_order() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    zeros = [0] * inst.n
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("ifft_tool",
            {"spectrum_re_x1000": [0] * inst.mask.size,
             "spectrum_im_x1000": [0] * inst.mask.size})]},
        {"tool_calls": [_tool_call_dict("compute_residual_tool",
            {"signal_x1000": zeros})]},
        {"tool_calls": [_tool_call_dict("sparsity_norm_tool",
            {"signal_x1000": zeros})]},
        _final_answer(inst),
    ])
    out = env.run_rollout_with_tools(solver, inst)
    assert out["meta"]["tool_sequence"] == [
        "ifft_tool", "compute_residual_tool", "sparsity_norm_tool",
    ]


def test_run_rollout_propagates_parse_failure_on_final() -> None:
    env = SparseFourierToolsEnv(conformal_quantile=2.0, max_tool_calls=5)
    inst = env.generate_instance(seed=0)
    solver = FakeLLMSolver([
        {"tool_calls": [_tool_call_dict("sparsity_norm_tool",
                                        {"signal_x1000": [0] * inst.n})]},
        "not JSON at all",
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


# ---------- Tool round-trip sanity ----------


def test_fft_round_trip_through_ifft_is_sane() -> None:
    """For truth sparse x, fft_tool gives y_hat; ifft_tool on y_hat returns a
    dense signal with nonzero L2 (not a no-op)."""
    inst = sf.generate_instance(seed=0)
    signal = _truth_dense_x1000(inst)
    fft_result = dispatch_tool("fft_tool", {"signal_x1000": signal}, inst)
    ifft_result = dispatch_tool(
        "ifft_tool",
        {"spectrum_re_x1000": fft_result["y_hat_re_x1000"],
         "spectrum_im_x1000": fft_result["y_hat_im_x1000"]},
        inst,
    )
    assert "signal_x1000" in ifft_result
    assert len(ifft_result["signal_x1000"]) == inst.n
    recovered = np.array(ifft_result["signal_x1000"]) / 1000.0
    assert float(np.linalg.norm(recovered)) > 0
