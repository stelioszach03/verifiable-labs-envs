"""Sandbox primitives for code-execution envs (Phase 24.B).

The single public surface is :func:`execute_in_sandbox` from
:mod:`verifiable_labs_envs.sandbox.code_execution_sandbox`. The
:class:`SandboxResult` dataclass and the :class:`TestManifest`
TypedDict describe the call-and-return shape that the
``code-humaneval`` / ``code-mini-repo`` env families consume.

PHASE_24_PLAN.md §6 locks the D2-A "subprocess + rlimit" mechanism
under a trusted-input scope (``/v1/score`` and ``/v1/datasets`` only).
The plan's upgrade-gate clause requires switching to D2-B (Docker)
or D2-C (Firecracker) before any public anonymous-submit endpoint
ships.
"""
from __future__ import annotations

from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MEM_BYTES,
    DEFAULT_NPROC,
    DEFAULT_NPROC_CAP,
    DEFAULT_TIMEOUT_S,
    DEFAULT_WALL_GRACE_S,
    SandboxResult,
    TestManifest,
    build_pytest_manifest,
    execute_in_sandbox,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MEM_BYTES",
    "DEFAULT_NPROC",
    "DEFAULT_NPROC_CAP",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_WALL_GRACE_S",
    "SandboxResult",
    "TestManifest",
    "build_pytest_manifest",
    "execute_in_sandbox",
    "execute_in_sandbox_sync",
    "parse_pytest_q_summary",
]
