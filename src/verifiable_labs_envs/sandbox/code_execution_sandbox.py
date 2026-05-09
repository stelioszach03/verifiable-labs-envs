"""Subprocess sandbox for code-execution envs (Phase 24.B).

PHASE_24_PLAN.md §6 spec, D2-A ruling. This module implements the
"subprocess + rlimit" sandbox: the customer-supplied test command
runs in a forked child whose ``RLIMIT_AS / CPU / NOFILE / NPROC /
FSIZE`` are pinned to the D5 values, with a wall-clock timeout
applied via ``Popen.communicate(timeout=...)``. On Linux the child
is launched under ``unshare -rn`` to enter a fresh user + network
namespace — outbound socket connect fails, defending the host
against arbitrary outbound traffic from customer code.

Trusted-input scope (D2-A locked, plan §5):

    The locked guarantee is *isolation between concurrent customer
    calls on the same Fly machine*, not *defence against a
    determined attacker who has compromised an API key*.

UPGRADE GATE — DO NOT REMOVE WITHOUT REVISING PHASE_24_PLAN.md:
the day a public anonymous "submit code, see reward" surface
enters roadmap, this primitive must flip to Docker (D2-B) or
Firecracker (D2-C). The sentinel test
``test_sandbox_upgrade_gate_documented`` enforces that this
docstring still mentions the gate.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

# ── D5 limits (locked) ────────────────────────────────────────────────

# 512 MB virtual address space cap. Defends against `bytearray(<<20)`
# style memory bombs while leaving plenty of room for the pytest +
# user-code workload (~150 MB resident at peak).
DEFAULT_MEM_BYTES: int = 512 * 1024 * 1024

# 30 s wall-clock — matches Phase 22 ``/v1/score`` timeout. Long
# enough for legitimate tests, short enough that the worker pool
# can't be wedged by a single slow customer.
DEFAULT_TIMEOUT_S: float = 30.0

# Grace seconds between SIGTERM and SIGKILL when the wall-clock
# timeout fires. SIGTERM lets pytest emit a final summary line if
# it can; SIGKILL guarantees forward progress.
DEFAULT_WALL_GRACE_S: float = 2.0

# 20 s of CPU time. RLIMIT_CPU is enforced by the kernel; the
# process gets SIGXCPU at the soft limit, then SIGKILL one second
# after the hard limit. We set soft = 20, hard = 22.
DEFAULT_CPU_SECONDS: int = 20

# Fork bomb defence. 16 child processes is enough for legitimate
# pytest workloads but kills `:(){ :|:& };:` style attacks.
DEFAULT_NPROC: int = 16
DEFAULT_NPROC_CAP: int = DEFAULT_NPROC  # alias for plan-doc symmetry

# fd-exhaustion defence. 64 fds is plenty for pytest + a handful of
# imports.
DEFAULT_NOFILE: int = 64

# 64 MB max created-file size — defends against `open("/tmp/x", "w").
# write("x" * 10**12)` style disk bombs.
DEFAULT_FSIZE_BYTES: int = 64 * 1024 * 1024

# Output truncation. 64 KB stdout + 64 KB stderr returned to caller.
DEFAULT_MAX_OUTPUT_BYTES: int = 64 * 1024


# ── Public dataclasses + types ────────────────────────────────────────


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed subprocess invocation.

    ``stdout`` / ``stderr`` are truncated to
    :data:`DEFAULT_MAX_OUTPUT_BYTES` before this object is
    constructed. ``timed_out`` indicates the wall-clock deadline
    expired (separate from CPU-seconds expiry, which surfaces as
    ``signal_terminated == signal.SIGXCPU``). ``oom_killed`` is a
    best-effort heuristic — set when the process was SIGKILLed
    without a preceding wall-clock timeout (either RLIMIT_AS or
    the kernel OOM killer fired).
    """

    exit_code: int
    stdout: str
    stderr: str
    wall_seconds: float
    timed_out: bool
    oom_killed: bool
    signal_terminated: int | None
    tmpdir_cleaned: bool


class TestManifest(TypedDict):
    """JSON manifest describing how to invoke the test runner.

    Phase 24 ships a single runner (``"pytest"``); the manifest shape
    is forward-compatible with v0.0.2 polyglot — adding ``"jest"`` /
    ``"go-test"`` won't break the env class or sandbox.
    """

    runner: Literal["pytest"]
    test_cmd: list[str]
    expected_exit: int
    timeout_s: float


# Stops pytest from trying to collect this TypedDict as a test class
# (``Test*`` prefix triggers heuristic collection — harmless warning).
TestManifest.__test__ = False  # type: ignore[attr-defined]


# ── Pytest summary parser ─────────────────────────────────────────────


# Matches the canonical pytest summary line:
#     "5 passed in 0.12s"
#     "3 failed, 2 passed in 0.34s"
#     "1 error, 4 passed, 1 skipped in 0.05s"
_PYTEST_SUMMARY = re.compile(
    r"(?P<count>\d+)\s+(?P<status>passed|failed|error|errors|skipped|deselected|xfailed|xpassed)"
)


def parse_pytest_q_summary(stdout: str) -> dict[str, int]:
    """Extract ``{passed, failed, error, ...}`` counts from pytest output.

    Returns zero counts when no summary line is found (e.g., the
    process was killed before pytest could write its tail). Callers
    should treat ``passed == 0 AND failed == 0`` as "no signal".
    """
    counts: dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    if not stdout:
        return counts
    # Walk every match; the last summary line wins (covers cases where
    # pytest prints multiple summaries, e.g. on collection error).
    for m in _PYTEST_SUMMARY.finditer(stdout):
        status = m.group("status")
        if status == "errors":
            status = "error"
        if status in counts:
            counts[status] = max(counts[status], int(m.group("count")))
    return counts


# ── Default factory ───────────────────────────────────────────────────


def build_pytest_manifest(
    test_files: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> TestManifest:
    """Produce the canonical ``pytest -q --tb=line`` manifest.

    The default invocation is deliberately tiny — coverage,
    parallelisation, and per-test timeouts are layered on top in the
    env modules that need them.
    """
    # ``sys.executable -m pytest`` instead of bare ``pytest`` so the
    # sandboxed subprocess doesn't depend on a ``pytest`` console
    # script being on ``$PATH``. Latent since 24.B (commit 665d26c) —
    # masked when CI Docker images installed pytest system-wide;
    # surfaces on every ``pip install --user`` setup. Diagnosis +
    # reproducer in ``reports/sandbox_investigation.md``.
    return {
        "runner": "pytest",
        "test_cmd": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            *test_files,
        ],
        "expected_exit": 0,
        "timeout_s": float(timeout_s),
    }


# ── Sandbox internals ─────────────────────────────────────────────────


def _set_rlimits(
    *,
    mem_bytes: int,
    cpu_seconds: int,
    nproc: int,
    nofile: int,
    fsize_bytes: int,
) -> None:
    """Apply D5 rlimits inside the forked child (Linux only).

    On non-Linux the ``resource`` module is still importable but the
    individual ``RLIMIT_*`` constants may be missing; we set what we
    can and silently skip the rest. The sandbox tests (which require
    every limit) run on Linux/WSL.
    """
    try:
        import resource  # noqa: PLC0415  (POSIX-only)
    except ImportError:  # pragma: no cover — Windows
        return

    def _try_set(name: str, value: tuple[int, int]) -> None:
        const = getattr(resource, name, None)
        if const is None:
            return
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(const, value)

    # RLIMIT_AS — virtual address space cap; the most reliable cap
    # on Python's allocations.
    _try_set("RLIMIT_AS", (mem_bytes, mem_bytes))
    # RLIMIT_CPU — CPU-seconds. Soft = cpu_seconds, hard = +2s grace.
    _try_set("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 2))
    # RLIMIT_NPROC — fork bomb defence.
    _try_set("RLIMIT_NPROC", (nproc, nproc))
    # RLIMIT_NOFILE — fd exhaustion defence.
    _try_set("RLIMIT_NOFILE", (nofile, nofile))
    # RLIMIT_FSIZE — disk bomb defence.
    _try_set("RLIMIT_FSIZE", (fsize_bytes, fsize_bytes))
    # RLIMIT_CORE = 0 — never write coredumps from sandboxed jobs.
    _try_set("RLIMIT_CORE", (0, 0))


def _preexec(
    *,
    mem_bytes: int,
    cpu_seconds: int,
    nproc: int,
    nofile: int,
    fsize_bytes: int,
) -> None:
    """``preexec_fn`` for :class:`subprocess.Popen` — runs in the child.

    Order: set process group (so ``os.killpg`` reaches the whole
    tree on timeout), then apply rlimits.
    """
    if hasattr(os, "setsid"):
        with contextlib.suppress(OSError):
            os.setsid()
    _set_rlimits(
        mem_bytes=mem_bytes,
        cpu_seconds=cpu_seconds,
        nproc=nproc,
        nofile=nofile,
        fsize_bytes=fsize_bytes,
    )


def _unshare_available() -> bool:
    """True when ``unshare -rn`` can be used for network isolation.

    Linux-only. Cached at module load via the lazy first call.
    """
    if sys.platform != "linux":
        return False
    return shutil.which("unshare") is not None


def _wrap_with_unshare(cmd: list[str]) -> list[str]:
    """Prefix ``cmd`` with ``unshare -rn --`` for net+user isolation.

    ``-r`` maps the calling user to root inside the new namespace
    (no privileged capabilities required outside the namespace).
    ``-n`` enters a fresh network namespace whose only interface is
    a down loopback — outbound connect fails immediately.
    """
    if not _unshare_available():
        return cmd
    return ["unshare", "-r", "-n", "--", *cmd]


def _truncate(data: bytes, limit: int) -> str:
    """UTF-8 decode + truncate to ``limit`` bytes (with marker)."""
    if data is None:
        return ""
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    head = data[:limit].decode("utf-8", errors="replace")
    return head + f"\n[... truncated, {len(data) - limit} bytes omitted ...]"


def _make_tmpdir() -> Path:
    """Allocate a fresh per-call sandbox directory under /tmp."""
    base = Path(tempfile.gettempdir()) / "vlabs-sandbox"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _write_files(tmpdir: Path, files: dict[str, str]) -> None:
    """Materialise ``{path: content}`` under ``tmpdir``.

    Refuses any path that escapes ``tmpdir`` (no leading ``/``, no
    ``..`` components surviving normalisation). This is defence in
    depth — the env code is supposed to hand us only relative paths.
    """
    for rel_path, content in files.items():
        candidate = (tmpdir / rel_path).resolve()
        if not str(candidate).startswith(str(tmpdir.resolve())):
            raise ValueError(
                f"refusing to write outside sandbox: {rel_path!r}"
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(content, encoding="utf-8")


# ── Public sandbox entry point ────────────────────────────────────────


def execute_in_sandbox_sync(
    *,
    files: dict[str, str],
    test_manifest: TestManifest,
    env_overrides: dict[str, str] | None = None,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    nproc: int = DEFAULT_NPROC,
    nofile: int = DEFAULT_NOFILE,
    fsize_bytes: int = DEFAULT_FSIZE_BYTES,
    wall_grace_s: float = DEFAULT_WALL_GRACE_S,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> SandboxResult:
    """Synchronous core of :func:`execute_in_sandbox`.

    Spawns the test command in a fresh tmpdir under D5 limits, waits
    up to ``test_manifest["timeout_s"]`` seconds, and tears down the
    tmpdir on every exit path. Used directly by tests that don't need
    an event loop; the async wrapper just runs this in a thread.
    """
    tmpdir = _make_tmpdir()
    timed_out = False
    oom_killed = False
    signal_terminated: int | None = None
    exit_code = -1
    stdout_bytes = b""
    stderr_bytes = b""
    tmpdir_cleaned = False
    start = time.monotonic()
    try:
        _write_files(tmpdir, files)

        cmd = _wrap_with_unshare(list(test_manifest["test_cmd"]))
        env = {**os.environ, **(env_overrides or {})}
        # Strip PYTHONPATH so the sandbox can't accidentally import
        # repo modules from outside the tmpdir.
        env.pop("PYTHONPATH", None)

        proc = subprocess.Popen(  # noqa: S603 — D2-A trusted-input scope
            cmd,
            cwd=str(tmpdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=lambda: _preexec(
                mem_bytes=mem_bytes,
                cpu_seconds=cpu_seconds,
                nproc=nproc,
                nofile=nofile,
                fsize_bytes=fsize_bytes,
            ),
            close_fds=True,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(
                timeout=test_manifest["timeout_s"]
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGTERM → grace → SIGKILL on the whole process group.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    timeout=wall_grace_s
                )
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                stdout_bytes, stderr_bytes = proc.communicate()
                exit_code = proc.returncode
        # Detect signal termination.
        if exit_code is not None and exit_code < 0:
            signal_terminated = -exit_code
            # SIGKILL without a wall-clock timeout → OOM (or rlimit fired).
            if signal_terminated == signal.SIGKILL and not timed_out:
                oom_killed = True
            # SIGXCPU is the kernel firing on RLIMIT_CPU.
            if signal_terminated == signal.SIGXCPU and not timed_out:
                # CPU-cap kill is also "non-OOM" but distinct; surface
                # via signal_terminated and leave oom_killed False.
                oom_killed = False
    finally:
        wall_seconds = time.monotonic() - start
        try:
            shutil.rmtree(tmpdir, ignore_errors=False)
            tmpdir_cleaned = True
        except OSError:
            # Best-effort fallback — partial cleanup, but never raise.
            shutil.rmtree(tmpdir, ignore_errors=True)
            tmpdir_cleaned = not tmpdir.exists()

    return SandboxResult(
        exit_code=int(exit_code if exit_code is not None else -1),
        stdout=_truncate(stdout_bytes or b"", max_output_bytes),
        stderr=_truncate(stderr_bytes or b"", max_output_bytes),
        wall_seconds=float(wall_seconds),
        timed_out=bool(timed_out),
        oom_killed=bool(oom_killed),
        signal_terminated=signal_terminated,
        tmpdir_cleaned=bool(tmpdir_cleaned),
    )


async def execute_in_sandbox(
    *,
    files: dict[str, str],
    test_manifest: TestManifest,
    env_overrides: dict[str, str] | None = None,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    nproc: int = DEFAULT_NPROC,
    nofile: int = DEFAULT_NOFILE,
    fsize_bytes: int = DEFAULT_FSIZE_BYTES,
    wall_grace_s: float = DEFAULT_WALL_GRACE_S,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> SandboxResult:
    """Async wrapper — runs :func:`execute_in_sandbox_sync` in a thread.

    Awaitable so the FastAPI handler stays non-blocking even when
    the underlying subprocess uses 30 s of wall-clock. The thread
    pool is the default ``asyncio`` executor; production deploys
    can swap in a sized pool via ``loop.set_default_executor``.
    """
    loop = asyncio.get_running_loop()

    def _run() -> SandboxResult:
        return execute_in_sandbox_sync(
            files=files,
            test_manifest=test_manifest,
            env_overrides=env_overrides,
            mem_bytes=mem_bytes,
            cpu_seconds=cpu_seconds,
            nproc=nproc,
            nofile=nofile,
            fsize_bytes=fsize_bytes,
            wall_grace_s=wall_grace_s,
            max_output_bytes=max_output_bytes,
        )

    return await loop.run_in_executor(None, _run)


__all__ = [
    "DEFAULT_CPU_SECONDS",
    "DEFAULT_FSIZE_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MEM_BYTES",
    "DEFAULT_NOFILE",
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
