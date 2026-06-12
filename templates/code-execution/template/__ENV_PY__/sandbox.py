"""Sandbox helper for __ENV_ID__.

Re-exports the platform-level sandbox primitive shipped in
``verifiable_labs_envs.sandbox``. Per-env scaffolds keep this thin
indirection so a customised sandbox (different rlimit defaults, a
different runner manifest factory) can be wired in without touching
the env / reward code.

PHASE_24_PLAN.md §6 locks the D2-A "subprocess + rlimit" mechanism
under a trusted-input scope. The upgrade-gate clause requires
switching to D2-B (Docker) or D2-C (Firecracker) before any public
anonymous-submit endpoint ships.
"""
from __future__ import annotations

from verifiable_labs_envs.sandbox import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MEM_BYTES,
    DEFAULT_NPROC,
    DEFAULT_TIMEOUT_S,
    SandboxResult,
    TestManifest,
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MEM_BYTES",
    "DEFAULT_NPROC",
    "DEFAULT_TIMEOUT_S",
    "SandboxResult",
    "TestManifest",
    "build_pytest_manifest",
    "execute_in_sandbox_sync",
    "parse_pytest_q_summary",
]
