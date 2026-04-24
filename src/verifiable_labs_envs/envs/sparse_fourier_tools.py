"""Environment 1c — tool-use sparse-Fourier recovery (primitive composition).

The LLM solves the standard sparse-Fourier problem but must *compose* ISTA
from five primitive operators rather than delegate to a solver oracle. No
single primitive is itself a reconstruction: the model has to iterate
``forward → residual → adjoint → threshold`` to converge.

Available tools:

- ``fft_tool(signal_x1000)``                 apply ``A = S·F`` to a length-n dense candidate; returns m measured coefficients.
- ``ifft_tool(spectrum_re_x1000, spectrum_im_x1000)``  zero-fill m coefficients at the mask + inverse DFT; returns a length-n real signal (the ``A^T`` adjoint of a measurement vector).
- ``threshold_tool(signal_x1000, tau_x1000)``  elementwise soft-threshold ``sign(x) · max(|x| − τ, 0)`` — the ISTA proximal step.
- ``compute_residual_tool(signal_x1000)``    returns ``r = y − A(x)`` as (re, im) arrays plus L2/max-abs; no oracle knowledge of the truth.
- ``sparsity_norm_tool(signal_x1000)``       returns ``‖x‖₁``, ``‖x‖₂`` and the count of nonzero entries above a small tolerance.

Older v0.1 names ``check_residual_tool`` and ``ista_tool`` are not exposed.
``check_residual_tool`` was renamed to ``compute_residual_tool`` for symmetry
with the dense-signal primitive set. ``ista_tool`` was an OMP oracle — any
model that called it copied the classical baseline verbatim. Keeping it
would have corrupted the "does the model actually compose ISTA" signal, so
it is deleted.

Reward is computed on the final parsed ``(support_idx, support_amp_x1000)``
answer, same as the single-turn env. Tool calls do not score directly —
they are the *means* by which the model converges to that answer.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from verifiable_labs_envs.envs.sparse_fourier import (
    DEFAULT_HYPERPARAMS as _BASE_DEFAULTS,
)
from verifiable_labs_envs.envs.sparse_fourier import (
    Instance,
    Prediction,
    SparseFourierEnv,
    _cached_quantile,
)
from verifiable_labs_envs.forward_ops import (
    sparse_fourier_adjoint,
    sparse_fourier_forward,
)
from verifiable_labs_envs.solvers.llm_solver import EnvAdapter, LLMSolver

NAME = "sparse-fourier-recovery-tools"
DEFAULT_MAX_TOOL_CALLS: int = 30
_NZ_TOL: float = 1e-3  # tolerance for "nonzero" in sparsity_norm_tool


# ──────────────────────────────────────────
# Tool schemas (OpenAI function-calling JSON)
# ──────────────────────────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fft_tool",
            "description": (
                "Apply A = S·F to a length-n dense candidate signal and return the m "
                "complex Fourier coefficients at the observed mask positions, scaled x1000 "
                "(real and imaginary parts, integer)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "length-n dense real signal, scaled x1000.",
                    },
                },
                "required": ["signal_x1000"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ifft_tool",
            "description": (
                "Zero-fill the given frequency coefficients at the observed mask positions "
                "and return the real inverse-DFT signal of length n (the adjoint A^T applied "
                "to a measurement vector)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spectrum_re_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "real parts x1000 at each of the m mask positions, in mask order.",
                    },
                    "spectrum_im_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "imaginary parts x1000 at each of the m mask positions.",
                    },
                },
                "required": ["spectrum_re_x1000", "spectrum_im_x1000"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "threshold_tool",
            "description": (
                "Elementwise soft-threshold operator: return sign(x) · max(|x| − τ, 0) for "
                "each entry of the input signal. This is the proximal step of ISTA/FISTA "
                "for an L1 regularizer. Entries with |x| ≤ τ become exactly zero."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "length-n real signal, scaled x1000.",
                    },
                    "tau_x1000": {
                        "type": "integer",
                        "description": "threshold τ scaled x1000; must be >= 0.",
                    },
                },
                "required": ["signal_x1000", "tau_x1000"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_residual_tool",
            "description": (
                "Given a length-n dense candidate signal, compute the Fourier residual "
                "r = y − A(x) and return its per-coefficient real/imag parts (x1000), L2 "
                "norm (x1000), and max-abs (x1000). Use this to monitor convergence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "length-n real signal, scaled x1000.",
                    },
                },
                "required": ["signal_x1000"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sparsity_norm_tool",
            "description": (
                "Return ||x||_1 (x1000), ||x||_2 (x1000), and the count of entries with "
                "|x| > 1e-3. Helpful for deciding when to shrink or grow the threshold τ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_x1000": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "length-n real signal, scaled x1000.",
                    },
                },
                "required": ["signal_x1000"],
                "additionalProperties": False,
            },
        },
    },
]


# ──────────────────────────────────────────
# Tool executors (pure Python, instance-bound at call time)
# ──────────────────────────────────────────


def _dense_from_args(args: dict[str, Any], instance: Instance) -> np.ndarray | dict[str, Any]:
    """Parse a dense length-n signal from a tool-call payload.

    Returns either the numpy array on success, or an error dict the caller
    should return directly. Length mismatch is the one failure the LLM will
    routinely hit; explicit is better than silent truncation here.
    """
    raw = args.get("signal_x1000", []) or []
    n = instance.n
    if len(raw) != n:
        return {"error": f"expected signal_x1000 of length {n}; got {len(raw)}"}
    x = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(raw):
        try:
            x[i] = float(v) / 1000.0
        except (TypeError, ValueError):
            return {"error": f"non-numeric entry at index {i}: {v!r}"}
    return x


def _tool_fft(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _dense_from_args(args, instance)
    if isinstance(x, dict):
        return x
    y_hat = sparse_fourier_forward(x.astype(np.complex128), instance.mask)
    return {
        "y_hat_re_x1000": [int(round(float(v) * 1000)) for v in y_hat.real.tolist()],
        "y_hat_im_x1000": [int(round(float(v) * 1000)) for v in y_hat.imag.tolist()],
    }


def _tool_ifft(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    re = args.get("spectrum_re_x1000", []) or []
    im = args.get("spectrum_im_x1000", []) or []
    m = instance.mask.size
    if len(re) != m or len(im) != m:
        return {
            "error": (
                f"expected {m} coefficients in each array; got re={len(re)}, im={len(im)}"
            )
        }
    try:
        coeffs = np.array(
            [float(r) / 1000.0 + 1j * float(i) / 1000.0 for r, i in zip(re, im, strict=False)],
            dtype=np.complex128,
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"non-numeric coefficient: {exc}"}
    z = sparse_fourier_adjoint(coeffs, instance.mask, instance.n)
    return {"signal_x1000": [int(round(float(v) * 1000)) for v in z.real.tolist()]}


def _tool_threshold(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _dense_from_args(args, instance)
    if isinstance(x, dict):
        return x
    tau_raw = args.get("tau_x1000")
    if tau_raw is None:
        return {"error": "missing required argument tau_x1000"}
    try:
        tau = float(tau_raw) / 1000.0
    except (TypeError, ValueError):
        return {"error": f"tau_x1000 must be numeric; got {tau_raw!r}"}
    if tau < 0:
        return {"error": f"tau_x1000 must be >= 0; got {tau_raw}"}
    shrunk = np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)
    return {
        "signal_x1000": [int(round(float(v) * 1000)) for v in shrunk.tolist()],
        "nonzero_count": int(np.sum(np.abs(shrunk) > _NZ_TOL)),
    }


def _tool_compute_residual(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _dense_from_args(args, instance)
    if isinstance(x, dict):
        return x
    y_hat = sparse_fourier_forward(x.astype(np.complex128), instance.mask)
    residual = np.asarray(instance.y) - np.asarray(y_hat)
    l2 = float(np.linalg.norm(residual))
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "residual_re_x1000": [int(round(float(v) * 1000)) for v in residual.real.tolist()],
        "residual_im_x1000": [int(round(float(v) * 1000)) for v in residual.imag.tolist()],
        "residual_l2_x1000": int(round(l2 * 1000)),
        "residual_max_abs_x1000": int(round(max_abs * 1000)),
    }


def _tool_sparsity_norm(args: dict[str, Any], instance: Instance) -> dict[str, Any]:
    x = _dense_from_args(args, instance)
    if isinstance(x, dict):
        return x
    l1 = float(np.sum(np.abs(x)))
    l2 = float(np.linalg.norm(x))
    nz = int(np.sum(np.abs(x) > _NZ_TOL))
    return {
        "l1_x1000": int(round(l1 * 1000)),
        "l2_x1000": int(round(l2 * 1000)),
        "nonzero_count": nz,
    }


_TOOL_DISPATCH = {
    "fft_tool": _tool_fft,
    "ifft_tool": _tool_ifft,
    "threshold_tool": _tool_threshold,
    "compute_residual_tool": _tool_compute_residual,
    "sparsity_norm_tool": _tool_sparsity_norm,
}


def dispatch_tool(name: str, arguments: str | dict, instance: Instance) -> dict[str, Any]:
    """Dispatch a tool call. ``arguments`` may be a JSON string (OpenAI format) or a dict."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"error": f"invalid JSON arguments: {exc}"}
    else:
        args = arguments or {}
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name!r}"}
    return handler(args, instance)


# ──────────────────────────────────────────
# Env
# ──────────────────────────────────────────


class SparseFourierToolsEnv(SparseFourierEnv):
    """``SparseFourierEnv`` with a primitive-composition tool-use rollout."""

    name: str = NAME

    def __init__(
        self,
        conformal_quantile: float,
        hyperparams: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        super().__init__(conformal_quantile, hyperparams, weights)
        if max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0; got {max_tool_calls}")
        self.max_tool_calls = int(max_tool_calls)

    def run_rollout_with_tools(
        self,
        solver: LLMSolver,
        instance: Instance,
        *,
        adapter: EnvAdapter | None = None,
        max_tool_calls: int | None = None,
        max_loops: int = 40,
    ) -> dict[str, Any]:
        """Run a tool-use rollout.

        The LLM is given the standard sparse-F prompt plus the 5 primitive tool
        schemas and must compose them into ISTA-like iterations. Each tool call
        is dispatched server-side via :func:`dispatch_tool`; its result is
        appended as a ``tool``-role message. After at most ``max_tool_calls``
        calls (or when the LLM emits no tool call), we parse the final JSON
        answer.

        ``max_loops`` bounds the outer while-loop as a safety net against
        runaway LLMs (e.g. a model that keeps emitting tool calls after being
        asked for a final answer).
        """
        from verifiable_labs_envs.solvers.llm_solver import (
            LLMSolverError,
            get_adapter,
        )

        if adapter is None:
            adapter = get_adapter(self.name)
        tool_cap = int(max_tool_calls if max_tool_calls is not None else self.max_tool_calls)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": adapter.system_prompt},
            {"role": "user", "content": adapter.build_user_prompt(instance)},
        ]
        tool_calls_total = 0
        tool_sequence: list[str] = []
        final_text: str | None = None

        for _ in range(max_loops):
            result = solver.complete_turns(messages, tools=TOOL_SCHEMAS)

            if result.tool_calls and tool_calls_total < tool_cap:
                remaining = tool_cap - tool_calls_total
                calls_this_turn = result.tool_calls[:remaining]
                messages.append({
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": calls_this_turn,
                })
                for call in calls_this_turn:
                    name = call.get("function", {}).get("name", "")
                    args = call.get("function", {}).get("arguments", "")
                    tool_result = dispatch_tool(name, args, instance)
                    tool_calls_total += 1
                    tool_sequence.append(name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(tool_result),
                    })
                continue

            if result.tool_calls and tool_calls_total >= tool_cap:
                messages.append({
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": result.tool_calls,
                })
                for call in result.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(
                            {"error": "tool-call cap reached; output final JSON answer now"}
                        ),
                    })
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool-call budget exhausted. Provide your final JSON answer now "
                        "with keys support_idx and support_amp_x1000. No tool calls."
                    ),
                })
                continue

            final_text = result.text or ""
            break

        if final_text is None:
            raise LLMSolverError(
                f"Tool-use rollout exited without a final answer after {max_loops} loops "
                f"(tool_calls={tool_calls_total})"
            )

        prediction: Prediction = adapter.parse_response(final_text, instance)
        scored = self.score(prediction, instance)
        scored["meta"] = {
            **scored["meta"],
            "tool_calls": tool_calls_total,
            "tool_sequence": tool_sequence,
            "max_tool_calls": tool_cap,
        }
        return scored


# ──────────────────────────────────────────
# Factory
# ──────────────────────────────────────────


def load_environment(
    calibration_quantile: float | None = None,
    *,
    fast: bool = True,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> SparseFourierToolsEnv:
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
    return SparseFourierToolsEnv(conformal_quantile=q, max_tool_calls=max_tool_calls)


__all__ = [
    "NAME",
    "DEFAULT_MAX_TOOL_CALLS",
    "TOOL_SCHEMAS",
    "SparseFourierToolsEnv",
    "dispatch_tool",
    "load_environment",
]
