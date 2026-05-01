"""Unit tests for ``vlabs_audit.runner`` — uses an injected fake ``runner_fn``."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from vlabs_audit.config import AuditConfig
from vlabs_audit.runner import (
    EpisodeOutput,
    EpisodeRunner,
    _augment_trace_with_cost,
    _parse_cost_usd,
    _parse_trace_path,
    default_episode_run,
)
from vlabs_audit.storage import AuditStore


def _make_cfg(envs: list[str], episodes: int = 3, seed_start: int = 0) -> AuditConfig:
    return AuditConfig(
        model="m",
        envs=envs,
        episodes=episodes,
        output=Path("/tmp/x.pdf"),
        seed_start=seed_start,
    )


def test_runner_executes_episodes_and_records_rewards(tmp_path: Path) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        calls.append((env, model, seed))
        jp = output_dir / f"{env}__{seed}.jsonl"
        jp.write_text("{}")
        return EpisodeOutput(reward=0.5 + 0.01 * seed, jsonl_path=jp)

    with AuditStore(db_path=tmp_path / "x.db") as store:
        runner = EpisodeRunner(store, parallel=2, runner_fn=fake_run)
        cfg = _make_cfg(["env-a", "env-b"], episodes=3, seed_start=10)
        aid = runner.run_audit(cfg, traces_dir=tmp_path / "traces")

        assert store.counts_by_status(aid) == {"success": 6}
        assert {(e, s) for (e, _, s) in calls} == {
            (env, seed) for env in ("env-a", "env-b") for seed in (10, 11, 12)
        }
        # All calls saw the same model passed via cfg.
        assert {m for (_, m, _) in calls} == {"m"}

        # Audit row is closed (finished_at populated).
        rec = store.get_audit(aid)
        assert rec is not None and rec.finished_at is not None

        # Per-row reward + jsonl path land in storage. seeds 10..12 →
        # rewards 0.60, 0.61, 0.62 from fake_run above.
        for r in store.list_runs(aid):
            assert r.reward is not None and 0.59 <= r.reward <= 0.63
            assert r.jsonl_path is not None and r.jsonl_path.endswith(".jsonl")


def test_runner_marks_failures_without_aborting_batch(tmp_path: Path) -> None:
    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        if seed == 1:
            raise RuntimeError("boom")
        jp = output_dir / f"{env}__{seed}.jsonl"
        jp.write_text("{}")
        return EpisodeOutput(reward=0.0, jsonl_path=jp)

    with AuditStore(db_path=tmp_path / "x.db") as store:
        runner = EpisodeRunner(store, parallel=1, runner_fn=fake_run)
        cfg = _make_cfg(["env-a"], episodes=3, seed_start=0)
        aid = runner.run_audit(cfg, traces_dir=tmp_path / "traces")

        assert store.counts_by_status(aid) == {"success": 2, "failed": 1}
        failed = [r for r in store.list_runs(aid) if r.status == "failed"]
        assert len(failed) == 1
        assert failed[0].seed == 1
        assert failed[0].error is not None
        assert "RuntimeError" in failed[0].error
        assert "boom" in failed[0].error


def test_runner_parallel_runs_concurrently(tmp_path: Path) -> None:
    """parallel=4 should overlap the four sleep(0.15) calls."""
    in_flight = {"max": 0, "now": 0}
    lock = threading.Lock()

    def slow_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        with lock:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
        time.sleep(0.15)
        with lock:
            in_flight["now"] -= 1
        jp = output_dir / f"{env}__{seed}.jsonl"
        jp.write_text("{}")
        return EpisodeOutput(reward=0.0, jsonl_path=jp)

    with AuditStore(db_path=tmp_path / "x.db") as store:
        runner = EpisodeRunner(store, parallel=4, runner_fn=slow_run)
        cfg = _make_cfg(["env"], episodes=4, seed_start=0)

        t0 = time.monotonic()
        aid = runner.run_audit(cfg, traces_dir=tmp_path / "traces")
        elapsed = time.monotonic() - t0

        assert store.counts_by_status(aid) == {"success": 4}
        # Strong concurrency check: at some point >=2 workers were active.
        assert in_flight["max"] >= 2, in_flight
        # Soft wallclock check: 4 × 0.15s serial = 0.6s; parallel=4 should be
        # well under that even on slow runners (allow generous slack).
        assert elapsed < 0.5, f"expected <0.5s with parallel=4, got {elapsed:.3f}"


def test_runner_resume_drains_stale_running_and_pending(tmp_path: Path) -> None:
    """Simulate a crash mid-flight: half the rows stuck in 'running'.

    ``resume_audit`` must reset the stale rows to ``pending`` and drain the
    full set, leaving everything in ``success``.
    """
    seen: list[int] = []

    def fake_run(env: str, model: str, seed: int, output_dir: Path) -> EpisodeOutput:
        seen.append(seed)
        jp = output_dir / f"{env}__{seed}.jsonl"
        jp.write_text("{}")
        return EpisodeOutput(reward=0.7, jsonl_path=jp)

    with AuditStore(db_path=tmp_path / "x.db") as store:
        # Manually scaffold the crashed-mid-flight state.
        aid = store.create_audit("m", {"envs": ["env-a"]})
        store.schedule_episodes(aid, "env-a", episodes=4, seed_start=0)
        all_runs = store.list_runs(aid)
        # 2 rows stuck in 'running', 2 still 'pending' (worker never picked them up).
        store.mark_running(all_runs[0].id)
        store.mark_running(all_runs[1].id)
        assert store.counts_by_status(aid) == {"running": 2, "pending": 2}

        runner = EpisodeRunner(store, parallel=2, runner_fn=fake_run)
        n = runner.resume_audit(aid, "m", traces_dir=tmp_path / "traces")
        assert n == 4
        assert store.counts_by_status(aid) == {"success": 4}
        assert sorted(seen) == [0, 1, 2, 3]


def test_runner_resume_unknown_audit_raises(tmp_path: Path) -> None:
    with AuditStore(db_path=tmp_path / "x.db") as store:
        runner = EpisodeRunner(
            store,
            runner_fn=lambda env, model, seed, output_dir: EpisodeOutput(
                reward=0.0, jsonl_path=output_dir / "x.jsonl"
            ),
        )
        with pytest.raises(ValueError, match="unknown audit_id"):
            runner.resume_audit("aud_does_not_exist", "m")


# ── default_episode_run subprocess parsing ────────────────────────────


def test_parse_cost_usd_handles_sdk_summary_line_formats() -> None:
    """The SDK prints either ``Cost: $X.XXXX`` (4 decimals) or ``$X.XX`` (2),
    or an em-dash for free / unpriced models."""
    # 4-decimal sub-cent
    out = "Time: 0m 03s · Cost: $0.0042\nTrace saved to /tmp/x.jsonl\n"
    assert _parse_cost_usd(out) == pytest.approx(0.0042)
    # 2-decimal cents-or-more
    out = "Time: 5m 12s · Cost: $0.42\n"
    assert _parse_cost_usd(out) == pytest.approx(0.42)
    out = "Time: 10m 00s · Cost: $1.50\n"
    assert _parse_cost_usd(out) == pytest.approx(1.50)
    # Em-dash means "no usage data" → None
    out = "Time: 0m 03s · Cost: —\n"
    assert _parse_cost_usd(out) is None
    # Stdout without any cost line (e.g. an early exit) → None
    assert _parse_cost_usd("nothing here\n") is None
    # Stray earlier 'Cost:' anywhere doesn't fool us — last one wins.
    out = (
        "spurious Cost: $9.99 noise on an earlier line\n"
        "Time: 1m 00s · Cost: $0.0024\n"
    )
    assert _parse_cost_usd(out) == pytest.approx(0.0024)


def test_augment_trace_with_cost_round_trips_existing_fields(tmp_path: Path) -> None:
    """``_augment_trace_with_cost`` adds the field without clobbering the rest."""
    p = tmp_path / "trace.jsonl"
    original = {
        "env_name": "env-a",
        "reward": 0.42,
        "parse_success": True,
        "reward_components": {"nmse": 0.3, "conformal": 1.0},
    }
    p.write_text(json.dumps(original) + "\n", encoding="utf-8")

    _augment_trace_with_cost(p, 0.0042)
    revived = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert revived["estimated_cost_usd"] == pytest.approx(0.0042)
    # Every original field is preserved.
    for k, v in original.items():
        assert revived[k] == v
    # File still ends in a newline; still a single JSONL line.
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(text.splitlines()) == 1


def test_parse_trace_path_picks_last_match() -> None:
    stdout = (
        "✓ Loading environment: env-a\n"
        "✓ Calibrating conformal interval (target 0.90)...\n"
        "✓ Running 1 episodes...\n"
        "\n"
        "Mean reward: 0.500 ± 0.000\n"
        "Coverage: 1.000 (target 0.90) ✓\n"
        "Time: 0m 02s · Cost: $0.0010\n"
        "\n"
        "Trace saved to /home/u/.verifiable/runs/env-a_claude-haiku_2026.jsonl\n"
    )
    p = _parse_trace_path(stdout)
    assert p == Path("/home/u/.verifiable/runs/env-a_claude-haiku_2026.jsonl")

    assert _parse_trace_path("nothing useful here\n") is None


@dataclass
class _FakeProc:
    """Stand-in for :class:`subprocess.CompletedProcess`."""

    returncode: int
    stdout: str
    stderr: str = ""


def test_default_episode_run_parses_subprocess_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end ``default_episode_run`` with subprocess + filesystem stubbed."""
    # 1) Stub `shutil.which` so the PATH check passes.
    import vlabs_audit.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/fake/verifiable")

    # 2) Pre-create the JSONL where verifiable would have written it.
    sdk_runs = tmp_path / "sdk_runs"
    sdk_runs.mkdir()
    sdk_jsonl = sdk_runs / "env-a_claude-haiku_2026.jsonl"
    sdk_jsonl.write_text('{"reward": 0.42, "seed": 7}\n')

    # 3) Stub subprocess.run to mimic the SDK's stdout layout, including the
    # `Cost: $X.XXXX` summary line so the augmenter has something to backfill.
    captured_cmd: list[list[str]] = []
    captured_homes: list[str] = []

    def fake_run(cmd, **kwargs):
        captured_cmd.append(list(cmd))
        sub_env = kwargs.get("env") or {}
        captured_homes.append(sub_env.get("HOME", ""))
        return _FakeProc(
            returncode=0,
            stdout=(
                "✓ Loading environment: env-a\n"
                "Mean reward: 0.420 ± 0.000\n"
                "Time: 0m 03s · Cost: $0.0024\n"
                f"Trace saved to {sdk_jsonl}\n"
            ),
        )

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    out_dir = tmp_path / "audit_out"
    result = default_episode_run("env-a", "claude-haiku", 7, out_dir)

    # Reward came from the JSONL.
    assert result.reward == pytest.approx(0.42)
    # File was relocated to the audit's output dir under a stable name.
    expected_dst = out_dir / "env-a__seed7.jsonl"
    assert result.jsonl_path == expected_dst
    assert expected_dst.exists()
    # Source file is gone (shutil.move).
    assert not sdk_jsonl.exists()
    # Cost was harvested from stdout and back-filled into the JSONL so
    # AuditStats.total_cost_usd sees real numbers downstream.
    revived = json.loads(expected_dst.read_text(encoding="utf-8").splitlines()[0])
    assert revived["estimated_cost_usd"] == pytest.approx(0.0024)
    assert revived["reward"] == pytest.approx(0.42)
    # CLI invocation does NOT pass --out (we relocate after the fact).
    assert captured_cmd == [
        [
            "verifiable", "run",
            "--env", "env-a",
            "--model", "claude-haiku",
            "--episodes", "1",
            "--seed", "7",
        ]
    ]
    # Each call gets its own HOME so the SDK's runs dir is unique per worker.
    assert len(captured_homes) == 1
    assert captured_homes[0] != ""
    assert captured_homes[0] != os.environ.get("HOME", "")


def test_default_episode_run_raises_when_verifiable_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vlabs_audit.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="verifiable.*PATH"):
        default_episode_run("env-a", "m", 0, tmp_path)


def test_default_episode_run_raises_on_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import vlabs_audit.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/fake/verifiable")
    monkeypatch.setattr(
        runner_mod.subprocess,
        "run",
        lambda cmd, **kw: _FakeProc(returncode=2, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="failed.*exit 2.*boom"):
        default_episode_run("env-a", "m", 0, tmp_path)


def test_default_episode_run_raises_on_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Surface ``subprocess.TimeoutExpired`` so the runner records 'failed'."""
    import vlabs_audit.runner as runner_mod

    monkeypatch.setattr(runner_mod.shutil, "which", lambda _name: "/fake/verifiable")

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))

    monkeypatch.setattr(runner_mod.subprocess, "run", boom)

    with pytest.raises(subprocess.TimeoutExpired):
        default_episode_run("env-a", "m", 0, tmp_path)
