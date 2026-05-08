"""Sandbox-primitive contract tests for __ENV_ID__.

The env's reward kernel relies on
:func:`verifiable_labs_envs.sandbox.execute_in_sandbox_sync`. These
contract tests verify the local re-export in ``__ENV_PY__.sandbox``
hands back the right surface; the platform-level sandbox isolation
suite lives in the parent repo's ``tests/test_sandbox.py``.
"""
from __future__ import annotations

import sys

import pytest

from __ENV_PY__.sandbox import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S,
    SandboxResult,
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="sandbox primitive requires Linux (rlimits + unshare).",
)


def test_sandbox_re_exports_match_platform_defaults():
    assert DEFAULT_MEM_BYTES == 512 * 1024 * 1024
    assert DEFAULT_TIMEOUT_S == 30.0


def test_build_pytest_manifest_shape():
    m = build_pytest_manifest(["test_solution.py"], timeout_s=10.0)
    assert m["runner"] == "pytest"
    assert m["test_cmd"][0] == "pytest"
    assert m["expected_exit"] == 0
    assert m["timeout_s"] == 10.0


def test_sandbox_smoke_passes_trivial_test():
    files = {"test_smoke.py": "def test_ok():\n    assert 1 + 1 == 2\n"}
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=build_pytest_manifest(["test_smoke.py"], timeout_s=10.0),
    )
    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    counts = parse_pytest_q_summary(result.stdout)
    assert counts["passed"] == 1
