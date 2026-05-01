"""Episode runner — shells out to the user-facing ``verifiable run`` CLI.

The runner is dependency-injectable: tests substitute a fake
``EpisodeRunFn`` that returns canned :class:`EpisodeOutput` instances
without spawning any subprocess. Production wiring uses
:func:`default_episode_run`, which expects the ``verifiable`` console
script (from ``packages/verifiable-labs``) to be on ``$PATH``.

Concurrency: a ``ThreadPoolExecutor`` with ``cfg.parallel`` workers.
Each task atomically transitions its row to ``running`` before invoking
the runner function, then marks ``success`` or ``failed`` on the way
out. Crashed prior runs are recovered via
:meth:`AuditStore.reset_stale_running` at the start of every audit.

Resume semantics: re-invoking ``run_audit`` with the same config (same
``audit_id`` returned to caller is fresh, but the underlying
``schedule_episodes`` call is idempotent on the unique
``(audit_id, env, episode_idx)``) is conceptually new. To resume an
interrupted run, use :meth:`EpisodeRunner.resume_audit(audit_id,
model)` which only drains the existing pending rows.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vlabs_audit.config import AuditConfig
from vlabs_audit.storage import AuditRunRecord, AuditStore, default_home

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeOutput:
    """Result of running a single (env, model, seed) episode."""

    reward: float
    jsonl_path: Path


EpisodeRunFn = Callable[[str, str, int, Path], EpisodeOutput]


def default_traces_dir(audit_id: str) -> Path:
    return default_home() / "traces" / audit_id


# ── default subprocess-based runner ──────────────────────────────────


_TRACE_SAVED_PREFIX = "Trace saved to "

# The SDK prints ``Time: 0m 03s · Cost: $0.0042`` (4 decimals < $0.01,
# else 2 decimals). When the agent did not surface usage data the SDK
# substitutes an em-dash (``Cost: —``); in that case we leave the trace
# without a cost field.
_COST_PATTERN = re.compile(r"Cost:\s*\$([0-9]+\.[0-9]+)")


def _parse_trace_path(stdout: str) -> Path | None:
    """Pull the JSONL output path out of ``verifiable run`` stdout.

    The SDK prints ``Trace saved to <path>`` on success. Walk the output
    bottom-up so a stray earlier line can't mislead us.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_TRACE_SAVED_PREFIX):
            return Path(line[len(_TRACE_SAVED_PREFIX):].strip())
    return None


def _parse_cost_usd(stdout: str) -> float | None:
    """Extract the per-invocation USD cost from the SDK summary line.

    Returns ``None`` when the SDK printed ``Cost: —`` (free / unpriced
    model). Single ``--episodes 1`` invocation, so the SDK's batch
    cost is also the per-episode cost.
    """
    for line in reversed(stdout.splitlines()):
        if "Cost:" not in line:
            continue
        m = _COST_PATTERN.search(line)
        return float(m.group(1)) if m else None
    return None


def _augment_trace_with_cost(jsonl_path: Path, cost_usd: float) -> None:
    """Re-write the single-line JSONL with ``estimated_cost_usd`` added.

    The SDK doesn't populate Trace.estimated_cost_usd today (only logs
    cost in the human-readable summary). We backfill the field here so
    :class:`vlabs_audit.stats.AuditStats.total_cost_usd` and the LaTeX
    cost-per-correct figure see real numbers downstream.
    """
    text = jsonl_path.read_text(encoding="utf-8")
    if not text.strip():
        return
    head, _, tail = text.partition("\n")
    record = json.loads(head)
    record["estimated_cost_usd"] = float(cost_usd)
    rewritten = json.dumps(record) + "\n"
    if tail:
        rewritten += tail
    jsonl_path.write_text(rewritten, encoding="utf-8")


def default_episode_run(
    env: str,
    model: str,
    seed: int,
    output_dir: Path,
) -> EpisodeOutput:
    """Shell out to ``verifiable run`` and parse the JSONL output.

    Invokes the modern provider-routed CLI:

    .. code-block:: text

        verifiable run --env <env> --model <model> --episodes 1 --seed <seed>

    The SDK writes the trace to ``~/.verifiable/runs/<env>_<model>_<ts>.jsonl``
    and prints ``Trace saved to <path>`` on stdout. We parse that line, then
    relocate the file under ``output_dir`` with a stable ``<env>__seed<n>.jsonl``
    name so the audit's traces land in one tidy directory.

    Raises :class:`RuntimeError` on non-zero exit, missing JSONL, or
    unparseable reward field.
    """
    if shutil.which("verifiable") is None:
        raise RuntimeError(
            "vlabs-audit: the `verifiable` command is not on PATH. "
            "Install the SDK with: pip install -e packages/verifiable-labs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "verifiable",
        "run",
        "--env",
        env,
        "--model",
        model,
        "--episodes",
        "1",
        "--seed",
        str(seed),
    ]

    # The SDK writes traces under ``Path.home() / ".verifiable" / "runs"``
    # with a second-precision timestamp filename. With ``parallel>=2``,
    # two workers running in the same wall-clock second collide on that
    # filename. Sandbox each subprocess in its own HOME so the runs dir
    # is unique per call. We keep ``PYTHONUSERBASE`` pointing at the
    # *original* user site-packages so the sandboxed Python can still
    # import the SDK installed via ``pip install --user``.
    original_home = os.environ.get("HOME") or os.path.expanduser("~")
    with tempfile.TemporaryDirectory(prefix="vlabs-audit-sdk-") as sdk_home:
        sub_env = os.environ.copy()
        sub_env["HOME"] = sdk_home
        sub_env.setdefault("PYTHONUSERBASE", os.path.join(original_home, ".local"))
        proc = subprocess.run(  # noqa: S603 — subprocess args fully controlled
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=sub_env,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(
                f"verifiable run --env {env} --seed {seed} failed "
                f"(exit {proc.returncode}): {tail}"
            )
        src_path = _parse_trace_path(proc.stdout)
        if src_path is None:
            raise RuntimeError(
                "verifiable run finished but did not print a 'Trace saved to ...' "
                f"line (stdout tail: {proc.stdout[-200:]!r})"
            )
        if not src_path.exists():
            raise RuntimeError(
                f"verifiable run reported {src_path} but the file does not exist."
            )
        dst = output_dir / f"{env}__seed{seed}.jsonl"
        shutil.move(str(src_path), str(dst))
        cost_usd = _parse_cost_usd(proc.stdout)
        if cost_usd is not None:
            _augment_trace_with_cost(dst, cost_usd)
    with dst.open() as fh:
        first_line = fh.readline().strip()
    if not first_line:
        raise RuntimeError(f"verifiable run produced empty JSONL at {dst}")
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"verifiable run JSONL is malformed: {exc}") from exc
    reward = record.get("reward")
    if reward is None:
        reward = record.get("score", 0.0)
    return EpisodeOutput(reward=float(reward), jsonl_path=dst)


# ── runner ───────────────────────────────────────────────────────────


class EpisodeRunner:
    """Drives a parallel batch of episodes against an :class:`AuditStore`."""

    def __init__(
        self,
        store: AuditStore,
        parallel: int = 4,
        runner_fn: EpisodeRunFn | None = None,
    ) -> None:
        self.store = store
        self.parallel = max(1, parallel)
        self.runner_fn = runner_fn or default_episode_run

    def run_audit(
        self,
        cfg: AuditConfig,
        traces_dir: Path | None = None,
    ) -> str:
        """Schedule all (env × episode) rows + drain them in parallel."""
        audit_id = self.store.create_audit(cfg.model, cfg.model_dump(mode="json"))
        for env in cfg.envs:
            self.store.schedule_episodes(
                audit_id, env, cfg.episodes, cfg.seed_start
            )
        target_dir = traces_dir or default_traces_dir(audit_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._drain(audit_id, cfg.model, target_dir)
        self.store.finish_audit(audit_id)
        return audit_id

    def resume_audit(
        self,
        audit_id: str,
        model: str,
        traces_dir: Path | None = None,
    ) -> int:
        """Pick up pending rows for an existing audit. Returns # processed."""
        audit = self.store.get_audit(audit_id)
        if audit is None:
            raise ValueError(f"unknown audit_id: {audit_id}")
        # If a previous run crashed mid-flight, those rows are stuck in
        # 'running'. Reset them to 'pending' so they get drained again.
        self.store.reset_stale_running(audit_id)
        target_dir = traces_dir or default_traces_dir(audit_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        return self._drain(audit_id, model, target_dir)

    # ── internals ─────────────────────────────────────────────────

    def _drain(self, audit_id: str, model: str, traces_dir: Path) -> int:
        pending = self.store.list_pending(audit_id)
        if not pending:
            return 0
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as ex:
            futures = {
                ex.submit(self._run_one, rec, model, traces_dir): rec
                for rec in pending
            }
            for fut in concurrent.futures.as_completed(futures):
                if fut.result():
                    completed += 1
        return completed

    def _run_one(
        self,
        rec: AuditRunRecord,
        model: str,
        traces_dir: Path,
    ) -> bool:
        self.store.mark_running(rec.id)
        try:
            out = self.runner_fn(rec.env, model, rec.seed, traces_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "vlabs_audit.episode_failed audit=%s env=%s seed=%d error=%s detail=%s",
                rec.audit_id,
                rec.env,
                rec.seed,
                type(exc).__name__,
                str(exc)[:200],
            )
            self.store.fail_episode(rec.id, f"{type(exc).__name__}: {exc}")
            return False
        self.store.complete_episode(rec.id, out.reward, out.jsonl_path)
        return True


__all__ = [
    "EpisodeOutput",
    "EpisodeRunFn",
    "EpisodeRunner",
    "_augment_trace_with_cost",
    "_parse_cost_usd",
    "default_episode_run",
    "default_traces_dir",
]
