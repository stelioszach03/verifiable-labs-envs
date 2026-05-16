"""scripts/check_pipeline_health.py — between-experiment health gate.

Run *between* experiments to confirm the pod is still fit for the next run:

1. Disk free above a threshold (default 10 GB).
2. GPU memory recovered (less than ``max_held_gb``, default 1 GB).
3. Conformal coverage drift across known envs below ``max_drift_frac``
   (default 0.05). Skipped if no prior coverage snapshot exists.
4. Last checkpoint loadable (skipped if no checkpoint exists yet).

Output: JSON report at ``reports/preflight/pipeline_health_{ts}.json``.

Exit:
  0 = all (non-skipped) checks ok
  1 = at least one check failed
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "preflight"

EXIT_OK: int = 0
EXIT_FAIL: int = 1


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    # "ok", "fail", or "skipped" — coverage / checkpoint checks may be skipped.
    status: str = "ok"


@dataclass
class HealthReport:
    started_at: float
    finished_at: float
    duration_s: float
    ok: bool
    exit_code: int
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "ok": self.ok,
            "exit_code": self.exit_code,
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Individual checks.
# ---------------------------------------------------------------------------
def check_disk_free(path: Path, min_gb: float) -> CheckResult:
    try:
        usage = shutil.disk_usage(str(path))
    except FileNotFoundError:
        return CheckResult(
            name="disk_free",
            ok=False,
            status="fail",
            detail=f"path missing: {path}",
        )
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        return CheckResult(
            name="disk_free",
            ok=False,
            status="fail",
            detail=f"{free_gb:.1f} GB free at {path} < required {min_gb:.1f} GB",
        )
    return CheckResult(
        name="disk_free",
        ok=True,
        detail=f"{free_gb:.1f} GB free at {path}",
    )


def check_gpu_memory_recovered(
    max_held_gb: float,
    *,
    torch_: Any | None = None,
) -> CheckResult:
    """Verify <= max_held_gb of GPU memory is currently allocated on device 0.

    Skipped (ok=True) if torch is missing or no CUDA device is present —
    the health check is not a hard gate on a CPU pod.
    """
    if torch_ is None:
        try:
            import torch as torch_real  # type: ignore
        except ImportError:
            return CheckResult(
                name="gpu_memory_recovered",
                ok=True,
                status="skipped",
                detail="torch not installed — skipped",
            )
        torch_ = torch_real

    if not torch_.cuda.is_available():
        return CheckResult(
            name="gpu_memory_recovered",
            ok=True,
            status="skipped",
            detail="no CUDA device — skipped",
        )

    try:
        allocated = torch_.cuda.memory_allocated(0)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(
            name="gpu_memory_recovered",
            ok=False,
            status="fail",
            detail=f"memory_allocated(0) raised: {exc!r}",
        )

    held_gb = float(allocated) / (1024 ** 3)
    if held_gb > max_held_gb:
        return CheckResult(
            name="gpu_memory_recovered",
            ok=False,
            status="fail",
            detail=f"{held_gb:.2f} GB held > limit {max_held_gb:.2f} GB",
        )
    return CheckResult(
        name="gpu_memory_recovered",
        ok=True,
        detail=f"{held_gb:.2f} GB held (<= {max_held_gb:.2f} GB)",
    )


def check_coverage_drift(
    baseline_path: Path,
    latest_path: Path,
    max_drift_frac: float,
) -> CheckResult:
    """Compare two coverage JSON snapshots: ``{env_id: float, ...}``.

    Both files must be present, parseable, and share at least one env id.
    """
    if not baseline_path.is_file() or not latest_path.is_file():
        return CheckResult(
            name="coverage_drift",
            ok=True,
            status="skipped",
            detail=(
                "no baseline / latest coverage snapshot — skipped "
                f"(baseline={baseline_path.name}, latest={latest_path.name})"
            ),
        )

    try:
        baseline = json.loads(baseline_path.read_text())
        latest = json.loads(latest_path.read_text())
    except json.JSONDecodeError as exc:
        return CheckResult(
            name="coverage_drift",
            ok=False,
            status="fail",
            detail=f"could not parse coverage snapshot: {exc}",
        )

    if not isinstance(baseline, dict) or not isinstance(latest, dict):
        return CheckResult(
            name="coverage_drift",
            ok=False,
            status="fail",
            detail="coverage snapshots must be JSON objects mapping env_id -> float",
        )

    drifts: dict[str, float] = {}
    for env_id, base_val in baseline.items():
        if env_id not in latest:
            continue
        try:
            drift = abs(float(latest[env_id]) - float(base_val))
        except (TypeError, ValueError):
            continue
        drifts[env_id] = drift

    if not drifts:
        return CheckResult(
            name="coverage_drift",
            ok=True,
            status="skipped",
            detail="no overlapping envs to compare — skipped",
        )

    worst_env = max(drifts, key=lambda k: drifts[k])
    worst = drifts[worst_env]
    if worst > max_drift_frac:
        return CheckResult(
            name="coverage_drift",
            ok=False,
            status="fail",
            detail=(
                f"worst drift {worst:.3f} on {worst_env} > limit "
                f"{max_drift_frac:.3f}"
            ),
        )
    return CheckResult(
        name="coverage_drift",
        ok=True,
        detail=f"worst drift {worst:.3f} on {worst_env} (<= {max_drift_frac:.3f})",
    )


def check_checkpoint_loadable(checkpoint_dir: Path) -> CheckResult:
    """Confirm the latest checkpoint dir contains a recognised, non-empty weight file.

    The deep ``torch.load`` is intentionally NOT done here — that requires
    paying the GPU memory + de-serialisation cost, which defeats the point
    of a between-experiment health gate. We only confirm an artefact exists.
    """
    if not checkpoint_dir.is_dir():
        return CheckResult(
            name="checkpoint_loadable",
            ok=True,
            status="skipped",
            detail=f"no checkpoint dir at {checkpoint_dir} — skipped",
        )

    candidates = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return CheckResult(
            name="checkpoint_loadable",
            ok=True,
            status="skipped",
            detail=f"{checkpoint_dir} empty — skipped",
        )

    latest = candidates[0]
    accept = (
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors",
    )
    found = [
        p.name
        for p in latest.iterdir()
        if p.name in accept and p.is_file() and p.stat().st_size > 0
    ]
    if not found:
        return CheckResult(
            name="checkpoint_loadable",
            ok=False,
            status="fail",
            detail=f"{latest} contains no recognised weight file",
        )
    return CheckResult(
        name="checkpoint_loadable",
        ok=True,
        detail=f"{latest.name} -> {', '.join(found)}",
    )


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run_health(
    *,
    disk_path: Path | None = None,
    min_disk_gb: float = 10.0,
    max_gpu_held_gb: float = 1.0,
    coverage_baseline: Path | None = None,
    coverage_latest: Path | None = None,
    max_drift_frac: float = 0.05,
    checkpoint_dir: Path | None = None,
    torch_: Any | None = None,
) -> HealthReport:
    started_at = time.time()

    checks: list[CheckResult] = []
    checks.append(check_disk_free(disk_path or REPO_ROOT, min_disk_gb))
    checks.append(check_gpu_memory_recovered(max_gpu_held_gb, torch_=torch_))
    checks.append(
        check_coverage_drift(
            coverage_baseline or (REPO_ROOT / "reports" / "coverage" / "baseline.json"),
            coverage_latest or (REPO_ROOT / "reports" / "coverage" / "latest.json"),
            max_drift_frac,
        )
    )
    checks.append(
        check_checkpoint_loadable(
            checkpoint_dir or (REPO_ROOT / "runs" / "checkpoints"),
        )
    )

    finished_at = time.time()
    ok = all(c.ok for c in checks)
    return HealthReport(
        started_at=started_at,
        finished_at=finished_at,
        duration_s=finished_at - started_at,
        ok=ok,
        exit_code=EXIT_OK if ok else EXIT_FAIL,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Between-experiment pipeline health check.",
    )
    parser.add_argument("--disk-path", type=Path, default=None)
    parser.add_argument("--min-disk-gb", type=float, default=10.0)
    parser.add_argument("--max-gpu-held-gb", type=float, default=1.0)
    parser.add_argument("--coverage-baseline", type=Path, default=None)
    parser.add_argument("--coverage-latest", type=Path, default=None)
    parser.add_argument("--max-drift-frac", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = run_health(
        disk_path=args.disk_path,
        min_disk_gb=args.min_disk_gb,
        max_gpu_held_gb=args.max_gpu_held_gb,
        coverage_baseline=args.coverage_baseline,
        coverage_latest=args.coverage_latest,
        max_drift_frac=args.max_drift_frac,
        checkpoint_dir=args.checkpoint_dir,
    )

    out_path = (
        args.output
        or DEFAULT_REPORTS_DIR / f"pipeline_health_{int(report.started_at)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))

    verdict = "OK" if report.ok else "FAIL"
    print(f"[health] {verdict}")
    for c in report.checks:
        if c.status == "skipped":
            marker = "[skip]"
        elif c.ok:
            marker = "[ok]"
        else:
            marker = "[fail]"
        print(f"  {marker} {c.name}: {c.detail}")
    print(f"  report -> {out_path}")

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
