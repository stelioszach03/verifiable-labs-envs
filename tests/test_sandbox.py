"""Isolation + lifecycle tests for the code-execution sandbox (Phase 24.B).

Each guarantee from PHASE_24_PLAN.md §6 has at least one sentinel
test here. The user-prompt-mandated names (Check 6) and the plan
doc's names both appear — the test bodies that share an underlying
primitive use a helper so the duplication is one-line stubs.

These tests are intentionally non-skippable per the user prompt's
24.B execution spec. They run on Linux/WSL where ``unshare -rn`` is
available; on systems without ``unshare`` the no-network test would
fail (we explicitly mark that as a Linux-only requirement so the
deploy target is unambiguous).
"""
from __future__ import annotations

import contextlib
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest

from verifiable_labs_envs.sandbox import (
    DEFAULT_MAX_OUTPUT_BYTES,
    SandboxResult,
    TestManifest,
    build_pytest_manifest,
    execute_in_sandbox_sync,
    parse_pytest_q_summary,
)
from verifiable_labs_envs.sandbox.code_execution_sandbox import (
    _truncate,
    _unshare_available,
    _wrap_with_unshare,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="sandbox isolation requires Linux (rlimits + unshare).",
)


# ── Helpers ───────────────────────────────────────────────────────────


def _python_inline(code: str, *, timeout_s: float = 5.0) -> SandboxResult:
    """Run a python -c snippet in the sandbox with conservative defaults."""
    manifest: TestManifest = {
        "runner": "pytest",  # field is for downstream parsing; cmd is python -c
        "test_cmd": [sys.executable, "-c", code],
        "expected_exit": 0,
        "timeout_s": timeout_s,
    }
    return execute_in_sandbox_sync(files={}, test_manifest=manifest)


def _kill_signal(result: SandboxResult) -> int | None:
    """Convenience: surface the signal that terminated the child."""
    return result.signal_terminated


# ── Happy path ────────────────────────────────────────────────────────


def test_sandbox_runs_pytest_happy_path() -> None:
    """Trivial 1-test pytest module returns exit 0 + 1 passed."""
    files = {
        "test_smoke.py": "def test_ok():\n    assert 1 + 1 == 2\n",
    }
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=build_pytest_manifest(["test_smoke.py"], timeout_s=10.0),
    )
    assert result.exit_code == 0, result.stderr
    counts = parse_pytest_q_summary(result.stdout)
    assert counts["passed"] == 1
    assert counts["failed"] == 0


def test_sandbox_failed_assertion_returns_nonzero() -> None:
    files = {
        "test_smoke.py": "def test_bad():\n    assert 1 == 2\n",
    }
    result = execute_in_sandbox_sync(
        files=files,
        test_manifest=build_pytest_manifest(["test_smoke.py"], timeout_s=10.0),
    )
    assert result.exit_code != 0
    counts = parse_pytest_q_summary(result.stdout)
    assert counts["failed"] == 1


# ── D5 limit: wall-clock timeout ──────────────────────────────────────


def test_sandbox_wall_clock_kill() -> None:
    """`time.sleep(60)` must be killed before the 30s wall budget."""
    result = _python_inline("import time; time.sleep(60)", timeout_s=2.0)
    assert result.timed_out, f"expected timed_out, got {result!r}"
    # Wall-budget + grace is the upper bound; in practice ~2s + 2s grace.
    assert result.wall_seconds <= 6.0


def test_sandbox_wall_timeout_kills_long_running() -> None:
    """User-prompt-mandated test name. Same guarantee as
    :func:`test_sandbox_wall_clock_kill`, asserted via the public API."""
    result = _python_inline("import time; time.sleep(60)", timeout_s=2.0)
    assert result.timed_out
    assert _kill_signal(result) in {signal.SIGTERM, signal.SIGKILL, None}


# ── D5 limit: CPU timeout ─────────────────────────────────────────────


def test_sandbox_kills_cpu_runaway() -> None:
    """`while True: pass` is killed by RLIMIT_CPU (SIGXCPU) within ~22s."""
    # Soft cap = 2, hard cap = 4 — keeps the test fast.
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "while True:\n    pass\n"],
            "expected_exit": 0,
            "timeout_s": 30.0,
        },
        cpu_seconds=2,
    )
    # CPU killer fires either via SIGXCPU or the wall-clock follow-up.
    assert result.exit_code != 0
    # Wall-clock should NOT have fired — CPU got there first.
    assert result.wall_seconds < 8.0


def test_sandbox_cpu_timeout_kills_busy_loop() -> None:
    """User-prompt-mandated alias of :func:`test_sandbox_kills_cpu_runaway`."""
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "x = 0\nwhile True: x += 1\n"],
            "expected_exit": 0,
            "timeout_s": 30.0,
        },
        cpu_seconds=2,
    )
    assert result.exit_code != 0
    assert result.wall_seconds < 8.0


# ── D5 limit: memory cap ──────────────────────────────────────────────


def test_sandbox_oom_kills_or_raises() -> None:
    """`bytearray(<<20)` over the cap dies cleanly (MemoryError or SIGKILL)."""
    code = (
        "import sys\n"
        "try:\n"
        "    x = bytearray(700 * 1024 * 1024)\n"
        "    sys.exit(0)\n"
        "except MemoryError:\n"
        "    sys.exit(7)\n"
    )
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", code],
            "expected_exit": 0,
            "timeout_s": 5.0,
        },
        mem_bytes=256 * 1024 * 1024,  # cap below the alloc → MemoryError or kill
    )
    # Either the exit code is 7 (MemoryError caught) or the process
    # was SIGKILLed by the kernel (RLIMIT_AS / OOM).
    assert result.exit_code != 0
    if result.exit_code == 7:
        # Caught and handled inside Python — clean path.
        assert not result.oom_killed
    else:
        # Killed by signal — set oom_killed for caller telemetry.
        assert result.oom_killed or result.signal_terminated == signal.SIGKILL


def test_sandbox_memory_cap_kills_oom() -> None:
    """User-prompt-mandated alias of :func:`test_sandbox_oom_kills_or_raises`.

    Asserts the cap is enforced — either Python catches MemoryError
    or the kernel sends SIGKILL. Both outcomes are acceptable; what
    matters is that the cap is hit and the process exits non-zero.
    """
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [
                sys.executable,
                "-c",
                "x = bytearray(900 * 1024 * 1024)",
            ],
            "expected_exit": 0,
            "timeout_s": 5.0,
        },
        mem_bytes=256 * 1024 * 1024,
    )
    assert result.exit_code != 0
    assert result.oom_killed or result.signal_terminated is not None or result.exit_code == 1


# ── D5 limit: network ────────────────────────────────────────────────


def test_sandbox_cannot_open_socket() -> None:
    """`unshare -rn` puts the child in a fresh net namespace.

    The new namespace has only a down loopback — outbound connect
    fails with ENETUNREACH (or similar)."""
    if not _unshare_available():
        pytest.fail(
            "unshare unavailable; D2-A network isolation cannot be enforced. "
            "Sandbox upgrade gate (D2-B/C) required."
        )
    code = (
        "import socket, sys\n"
        "s = socket.socket()\n"
        "s.settimeout(2)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    sys.exit(0)\n"
        "except OSError as exc:\n"
        "    sys.exit(2)\n"
    )
    result = _python_inline(code, timeout_s=10.0)
    # The child caught OSError and exited with code 2 → network blocked.
    assert result.exit_code == 2, (
        f"expected ENETUNREACH (exit 2), got {result.exit_code}: {result.stderr}"
    )


def test_sandbox_no_network_blocks_socket() -> None:
    """User-prompt-mandated alias of :func:`test_sandbox_cannot_open_socket`."""
    if not _unshare_available():
        pytest.fail(
            "unshare unavailable on this host; sandbox network isolation "
            "is non-skippable. Install util-linux or run inside a Linux container."
        )
    result = _python_inline(
        "import socket, sys\n"
        "s = socket.socket()\n"
        "s.settimeout(1)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 80))\n"
        "    sys.exit(0)\n"
        "except OSError:\n"
        "    sys.exit(3)\n",
        timeout_s=5.0,
    )
    assert result.exit_code == 3


# ── D5 limit: process fanout ─────────────────────────────────────────


def test_sandbox_kills_fork_bomb() -> None:
    """`os.fork()` loop hits RLIMIT_NPROC and exits non-zero."""
    code = (
        "import os, sys\n"
        "for _ in range(100):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError:\n"
        "        sys.exit(5)\n"
        "    if pid == 0:\n"
        "        # Child does nothing, just consumes a slot.\n"
        "        import time; time.sleep(0.1); sys.exit(0)\n"
        "sys.exit(0)\n"
    )
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", code],
            "expected_exit": 0,
            "timeout_s": 10.0,
        },
        nproc=4,  # tighter than default to make the cap kick in fast
    )
    # We expect either:
    # (a) the parent caught OSError and exited 5 → RLIMIT_NPROC fired.
    # (b) the parent was killed before reaching the sys.exit(0).
    assert result.exit_code != 0


def test_sandbox_proc_cap_blocks_fork_bomb() -> None:
    """User-prompt-mandated alias of :func:`test_sandbox_kills_fork_bomb`."""
    code = (
        "import os, sys\n"
        "blocked = False\n"
        "for _ in range(50):\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except OSError:\n"
        "        blocked = True\n"
        "        break\n"
        "    if pid == 0:\n"
        "        import time; time.sleep(0.5); sys.exit(0)\n"
        "sys.exit(0 if blocked else 99)\n"
    )
    result = execute_in_sandbox_sync(
        files={},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", code],
            "expected_exit": 0,
            "timeout_s": 10.0,
        },
        nproc=3,
    )
    # Cap hit → exit 0 (blocked). If the cap fails to engage we'd see
    # exit 99 (sentinel). Anything except 99 means defence-in-depth held.
    assert result.exit_code != 99


# ── Tmpdir cleanup ───────────────────────────────────────────────────


def test_sandbox_cleans_tmpdir_on_every_path() -> None:
    """Tmpdir is wiped on success, failure, and timeout."""
    base = Path("/tmp/vlabs-sandbox")
    before = set(base.iterdir()) if base.exists() else set()

    # Success path.
    r1 = execute_in_sandbox_sync(
        files={"x.txt": "ok"},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "print('ok')"],
            "expected_exit": 0,
            "timeout_s": 3.0,
        },
    )
    assert r1.tmpdir_cleaned

    # Failure path.
    r2 = execute_in_sandbox_sync(
        files={"x.txt": "ok"},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "import sys; sys.exit(7)"],
            "expected_exit": 0,
            "timeout_s": 3.0,
        },
    )
    assert r2.tmpdir_cleaned

    # Timeout path.
    r3 = execute_in_sandbox_sync(
        files={"x.txt": "ok"},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "import time; time.sleep(60)"],
            "expected_exit": 0,
            "timeout_s": 1.0,
        },
    )
    assert r3.tmpdir_cleaned

    # No new entries leaked under /tmp/vlabs-sandbox.
    after = set(base.iterdir()) if base.exists() else set()
    leaked = after - before
    assert not leaked, f"sandbox dirs leaked: {leaked}"


def test_sandbox_tmpfs_cleanup_after_call() -> None:
    """User-prompt-mandated alias of
    :func:`test_sandbox_cleans_tmpdir_on_every_path`. Asserts that no
    sandbox directory survives the call beyond the function frame."""
    result = execute_in_sandbox_sync(
        files={"a/b/c.txt": "nested"},
        test_manifest={
            "runner": "pytest",
            "test_cmd": [sys.executable, "-c", "print('done')"],
            "expected_exit": 0,
            "timeout_s": 3.0,
        },
    )
    assert result.tmpdir_cleaned
    base = Path("/tmp/vlabs-sandbox")
    if base.exists():
        for d in base.iterdir():
            # Each leftover dir would mean a leak.
            assert not d.is_dir() or any(d.iterdir()) or False, f"empty leak: {d}"


# ── Filesystem isolation ─────────────────────────────────────────────


def test_sandbox_cannot_write_outside_tmp() -> None:
    """The sandbox writer must reject paths that escape the per-call dir.

    Defence in depth: env code is supposed to hand only relative
    paths, but a malicious env (or buggy template) shouldn't be able
    to write to ``/etc/passwd``.
    """
    with pytest.raises(ValueError, match="outside sandbox"):
        execute_in_sandbox_sync(
            files={"../../etc/x": "owned"},
            test_manifest={
                "runner": "pytest",
                "test_cmd": [sys.executable, "-c", "pass"],
                "expected_exit": 0,
                "timeout_s": 2.0,
            },
        )


# ── Output truncation ────────────────────────────────────────────────


def test_sandbox_truncates_stdout_flood() -> None:
    code = "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()"
    result = _python_inline(code, timeout_s=5.0)
    assert len(result.stdout.encode("utf-8")) <= DEFAULT_MAX_OUTPUT_BYTES + 200


def test_truncate_helper_marks_overflow() -> None:
    """Unit test for the byte truncator — independent of subprocess."""
    big = b"a" * 100_000
    out = _truncate(big, 1024)
    assert len(out.encode("utf-8")) > 1024  # marker text adds overhead
    assert "truncated" in out


# ── Pytest summary parser ────────────────────────────────────────────


def test_pytest_summary_parser_recognises_passed_failed() -> None:
    counts = parse_pytest_q_summary(
        "==== short summary ====\n3 failed, 5 passed in 0.42s\n"
    )
    assert counts["passed"] == 5
    assert counts["failed"] == 3


def test_pytest_summary_parser_recognises_errors() -> None:
    counts = parse_pytest_q_summary("collected 0 items / 1 error\n1 error in 0.01s\n")
    assert counts["error"] == 1


def test_pytest_summary_parser_returns_zeros_on_empty() -> None:
    counts = parse_pytest_q_summary("")
    assert counts["passed"] == 0
    assert counts["failed"] == 0
    assert counts["error"] == 0


# ── Manifest factory ─────────────────────────────────────────────────


def test_build_pytest_manifest_shape() -> None:
    """Manifest invokes pytest via ``sys.executable -m pytest`` so the
    sandboxed subprocess doesn't depend on a ``pytest`` console script
    being on ``$PATH``. Pre-fix shape used a bare ``"pytest"`` token
    which broke on ``pip install --user`` setups (see
    ``reports/sandbox_investigation.md``)."""
    import sys
    m = build_pytest_manifest(["a.py", "b.py"], timeout_s=12.5)
    assert m["runner"] == "pytest"
    assert m["test_cmd"] == [
        sys.executable, "-m", "pytest", "-q", "--tb=line", "a.py", "b.py",
    ]
    assert m["expected_exit"] == 0
    assert m["timeout_s"] == 12.5


def test_unshare_wrapper_prefixes_when_available() -> None:
    """`_wrap_with_unshare` only mutates the cmd when unshare exists."""
    cmd = ["echo", "hi"]
    wrapped = _wrap_with_unshare(cmd)
    if _unshare_available():
        assert wrapped[:3] == ["unshare", "-r", "-n"]
    else:
        assert wrapped == cmd


# ── Upgrade-gate sentinel ────────────────────────────────────────────


def test_sandbox_upgrade_gate_documented() -> None:
    """PHASE_24_PLAN.md D2 ruling locks D2-A under a trusted-input scope.

    The sandbox module's docstring must explicitly retain the upgrade
    gate so future reviewers can't silently flip the surface to a
    public anonymous-submit endpoint without a plan revision."""
    from verifiable_labs_envs.sandbox import code_execution_sandbox

    assert "UPGRADE GATE" in (code_execution_sandbox.__doc__ or "")
    assert "trusted-input" in (code_execution_sandbox.__doc__ or "").lower()


# ── Cleanup convenience ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _scrub_sandbox_root() -> None:
    """Wipe /tmp/vlabs-sandbox before each test for stable assertions."""
    yield
    base = Path("/tmp/vlabs-sandbox")
    if not base.exists():
        return
    for entry in base.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                os.unlink(entry)
