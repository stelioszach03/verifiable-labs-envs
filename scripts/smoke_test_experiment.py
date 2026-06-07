"""scripts/smoke_test_experiment.py — Phase 18 / E1-E10 smoke test orchestrator.

Validates the wiring of a planned experiment BEFORE Stelios spends real GPU
time on a full training run. Performs:

1. Experiment ID resolution against the registry below.
2. Python dependency check (torch + the per-experiment heavy deps).
3. GPU availability check (CUDA device, advertised VRAM >= per-experiment min).
4. Disk-free check on the working volume.
5. Optional dry-run probe (``--dry-run-after``): import the experiment runner
   module and confirm the callable exists — never invokes training.

Writes a JSON report at ``reports/preflight/smoke_{experiment_id}.json``.

Exit codes:
  0 = all checks pass — safe to fire the full run
  1 = setup failure (GPU missing, dependency missing, disk full, unknown experiment)
  2 = reward signal zero  (reserved — set by *post-training* smoke variants)
  3 = OOM                 (reserved — set by *actual-training* smoke variants)
  4 = JSON parse failure  (reserved — set by vLLM guided-decoding smoke)

Target cost: < $0.30 per smoke test on RTX 5090 ($0.99/hr -> ~18 min budget).

This is the bug-net before any Phase 18 / E1-E10 real experiment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "preflight"

# ---------------------------------------------------------------------------
# Exit codes (module-level for unit-test access).
# ---------------------------------------------------------------------------
EXIT_OK: int = 0
EXIT_SETUP: int = 1
EXIT_REWARD_ZERO: int = 2
EXIT_OOM: int = 3
EXIT_JSON_FAIL: int = 4


# ---------------------------------------------------------------------------
# Experiment registry.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentSpec:
    """Static metadata for a planned experiment.

    ``runner_module`` / ``runner_callable`` are consumed by ``--dry-run-after``
    to confirm the runner is importable. For experiments whose runner is not
    yet authored, set ``runner_module=None`` and the smoke test marks the
    runner-import check as *deferred* (pass) so the orchestration can land
    additively without coupling to runner authoring.
    """

    experiment_id: str
    description: str
    runner_module: str | None
    runner_callable: str | None
    required_deps: tuple[str, ...]
    min_vram_gb: float
    min_disk_gb: float
    estimated_full_cost_usd: float


# Per the Phase-18 prep protocol — Phase 18 redo + E1..E10 ablations
# (E9 is intentionally skipped: the plan has no E9 slot).
EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "phase18-redo": ExperimentSpec(
        experiment_id="phase18-redo",
        description="Phase 18 redo — GRPO on Qwen-1.5B with calibrated rewards",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=20.0,
        estimated_full_cost_usd=7.0,
    ),
    "E1": ExperimentSpec(
        experiment_id="E1",
        description="E1 — Replicate Phase 13.2 baseline on sparse-fourier-recovery",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=20.0,
        estimated_full_cost_usd=5.0,
    ),
    "E2": ExperimentSpec(
        experiment_id="E2",
        description="E2 — Multi-env training (3 envs simultaneously)",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=30.0,
        estimated_full_cost_usd=12.0,
    ),
    "E3": ExperimentSpec(
        experiment_id="E3",
        description="E3 — Cross-domain (train on signal envs, eval on math held-out)",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=25.0,
        estimated_full_cost_usd=15.0,
    ),
    "E4": ExperimentSpec(
        experiment_id="E4",
        description="E4 — Baseline reward model: Qwen-1.5B student produces calibrated scores",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "trl", "peft", "accelerate"),
        min_vram_gb=20.0,
        min_disk_gb=20.0,
        estimated_full_cost_usd=8.0,
    ),
    "E5a": ExperimentSpec(
        experiment_id="E5a",
        description="E5a — Dataset scaling (5k samples)",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "trl", "peft", "accelerate"),
        min_vram_gb=20.0,
        min_disk_gb=20.0,
        estimated_full_cost_usd=8.0,
    ),
    "E5b": ExperimentSpec(
        experiment_id="E5b",
        description="E5b — Dataset scaling (15k samples)",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "trl", "peft", "accelerate"),
        min_vram_gb=20.0,
        min_disk_gb=25.0,
        estimated_full_cost_usd=9.0,
    ),
    "E6": ExperimentSpec(
        experiment_id="E6",
        description="E6 — Teacher ablation: env-derived vs frontier-model labels",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=25.0,
        estimated_full_cost_usd=15.0,
    ),
    "E7": ExperimentSpec(
        experiment_id="E7",
        description="E7 — Student size: Llama-3B (memory headroom check)",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate", "bitsandbytes"),
        min_vram_gb=28.0,
        min_disk_gb=25.0,
        estimated_full_cost_usd=12.0,
    ),
    "E8": ExperimentSpec(
        experiment_id="E8",
        description="E8 — Final benchmark with best config from E1-E7",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=20.0,
        estimated_full_cost_usd=8.0,
    ),
    "E10": ExperimentSpec(
        experiment_id="E10",
        description="E10 — Adversarial robustness probes",
        runner_module=None,
        runner_callable=None,
        required_deps=("torch", "vllm", "trl", "peft", "accelerate"),
        min_vram_gb=24.0,
        min_disk_gb=15.0,
        estimated_full_cost_usd=3.0,
    ),
}

ALLOWED_EXPERIMENTS: tuple[str, ...] = tuple(sorted(EXPERIMENT_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Helpers / data classes.
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """One named check with pass/fail + a human-readable detail."""

    name: str
    ok: bool
    detail: str


@dataclass
class SmokeReport:
    experiment_id: str
    started_at: float
    finished_at: float
    duration_s: float
    ok: bool
    exit_code: int
    checks: list[CheckResult] = field(default_factory=list)
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "ok": self.ok,
            "exit_code": self.exit_code,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "checks": [asdict(c) for c in self.checks],
        }


def estimate_cost_usd(duration_s: float, hourly_rate_usd: float = 0.99) -> float:
    """RunPod-style hourly billing. RTX 5090 is ~$0.99/hr in May-2026."""
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s}")
    return (duration_s / 3600.0) * hourly_rate_usd


# ---------------------------------------------------------------------------
# Individual checks. Each returns a CheckResult.
# ---------------------------------------------------------------------------
def check_experiment_registered(
    experiment_id: str,
    registry: dict[str, ExperimentSpec] | None = None,
) -> CheckResult:
    """Confirm the experiment ID is in the registry."""
    reg = registry if registry is not None else EXPERIMENT_REGISTRY
    if experiment_id in reg:
        spec = reg[experiment_id]
        return CheckResult(
            name="experiment_registered",
            ok=True,
            detail=f"{experiment_id}: {spec.description}",
        )
    return CheckResult(
        name="experiment_registered",
        ok=False,
        detail=(
            f"unknown experiment {experiment_id!r}; allowed: "
            + ", ".join(sorted(reg.keys()))
        ),
    )


def check_dependencies(deps: tuple[str, ...]) -> CheckResult:
    """Check each dependency is importable via importlib.util.find_spec."""
    missing: list[str] = []
    for dep in deps:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    if missing:
        return CheckResult(
            name="dependencies",
            ok=False,
            detail=f"missing: {', '.join(missing)} (need: {', '.join(deps) or '(none)'})",
        )
    return CheckResult(
        name="dependencies",
        ok=True,
        detail=f"all present: {', '.join(deps) if deps else '(none)'}",
    )


def check_gpu_available(
    min_vram_gb: float,
    *,
    torch_: Any | None = None,
) -> CheckResult:
    """CUDA device 0 must exist with VRAM >= min_vram_gb.

    ``torch_`` lets tests inject a stub. When ``None``, the real ``torch``
    module is imported lazily — so the surrounding script imports cleanly on
    machines without torch installed.
    """
    if torch_ is None:
        try:
            import torch as torch_real  # type: ignore
        except ImportError:
            return CheckResult(
                name="gpu_available",
                ok=False,
                detail="torch not installed",
            )
        torch_ = torch_real

    if not torch_.cuda.is_available():
        return CheckResult(
            name="gpu_available",
            ok=False,
            detail="torch.cuda.is_available() is False",
        )

    try:
        props = torch_.cuda.get_device_properties(0)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(
            name="gpu_available",
            ok=False,
            detail=f"could not query device 0: {exc}",
        )

    vram_gb = float(getattr(props, "total_memory", 0)) / (1024 ** 3)
    if vram_gb < min_vram_gb:
        return CheckResult(
            name="gpu_available",
            ok=False,
            detail=(
                f"VRAM {vram_gb:.1f} GB < required {min_vram_gb:.1f} GB on "
                f"{getattr(props, 'name', 'unknown')}"
            ),
        )
    return CheckResult(
        name="gpu_available",
        ok=True,
        detail=f"{getattr(props, 'name', 'unknown')} ({vram_gb:.1f} GB)",
    )


def check_disk_free(path: Path, min_gb: float) -> CheckResult:
    """Path must have at least ``min_gb`` of free space."""
    try:
        usage = shutil.disk_usage(str(path))
    except FileNotFoundError:
        return CheckResult(
            name="disk_free",
            ok=False,
            detail=f"path does not exist: {path}",
        )
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        return CheckResult(
            name="disk_free",
            ok=False,
            detail=f"{free_gb:.1f} GB free at {path} < required {min_gb:.1f} GB",
        )
    return CheckResult(
        name="disk_free",
        ok=True,
        detail=f"{free_gb:.1f} GB free at {path}",
    )


def check_runner_importable(
    runner_module: str | None,
    runner_callable: str | None,
) -> CheckResult:
    """Confirm the experiment runner is wired.

    If ``runner_module`` is None, the runner has not been authored yet —
    return ok=True with a *deferred* detail so this gate can be enabled
    additively as runners come online.
    """
    if runner_module is None:
        return CheckResult(
            name="runner_importable",
            ok=True,
            detail="runner not wired yet — deferred",
        )

    if importlib.util.find_spec(runner_module) is None:
        return CheckResult(
            name="runner_importable",
            ok=False,
            detail=f"runner module not importable: {runner_module}",
        )

    if runner_callable is None:
        return CheckResult(
            name="runner_importable",
            ok=True,
            detail=f"module importable: {runner_module} (no callable specified)",
        )

    try:
        mod = importlib.import_module(runner_module)
    except Exception as exc:  # noqa: BLE001 - bubble import errors up
        return CheckResult(
            name="runner_importable",
            ok=False,
            detail=f"import {runner_module} raised: {exc!r}",
        )

    fn = getattr(mod, runner_callable, None)
    if fn is None:
        return CheckResult(
            name="runner_importable",
            ok=False,
            detail=f"{runner_module}.{runner_callable} not found",
        )
    if not callable(fn):
        return CheckResult(
            name="runner_importable",
            ok=False,
            detail=(
                f"{runner_module}.{runner_callable} is not callable "
                f"({type(fn).__name__})"
            ),
        )
    return CheckResult(
        name="runner_importable",
        ok=True,
        detail=f"{runner_module}.{runner_callable} callable",
    )


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run_smoke(
    experiment_id: str,
    *,
    max_steps: int = 5,
    dry_run_after: bool = False,
    disk_check_path: Path | None = None,
    registry: dict[str, ExperimentSpec] | None = None,
    hourly_rate_usd: float = 0.99,
    torch_: Any | None = None,
) -> SmokeReport:
    """Run the smoke pipeline + return a structured report.

    This function does **not** write JSON to disk — the CLI ``main`` does that.
    Keeping the pure logic side-effect-free makes unit testing trivial.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    started_at = time.time()
    reg = registry if registry is not None else EXPERIMENT_REGISTRY

    checks: list[CheckResult] = []

    # 1. Experiment registered.
    reg_check = check_experiment_registered(experiment_id, registry=reg)
    checks.append(reg_check)
    if not reg_check.ok:
        finished_at = time.time()
        duration = finished_at - started_at
        return SmokeReport(
            experiment_id=experiment_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration,
            ok=False,
            exit_code=EXIT_SETUP,
            checks=checks,
            estimated_cost_usd=estimate_cost_usd(duration, hourly_rate_usd),
        )

    spec = reg[experiment_id]

    # 2. Dependencies.
    checks.append(check_dependencies(spec.required_deps))

    # 3. GPU.
    checks.append(check_gpu_available(spec.min_vram_gb, torch_=torch_))

    # 4. Disk free.
    disk_path = disk_check_path or REPO_ROOT
    checks.append(check_disk_free(disk_path, spec.min_disk_gb))

    # 5. Runner importable — only if explicitly requested.
    if dry_run_after:
        checks.append(
            check_runner_importable(spec.runner_module, spec.runner_callable)
        )

    finished_at = time.time()
    duration = finished_at - started_at
    ok = all(c.ok for c in checks)

    return SmokeReport(
        experiment_id=experiment_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration,
        ok=ok,
        exit_code=EXIT_OK if ok else EXIT_SETUP,
        checks=checks,
        estimated_cost_usd=estimate_cost_usd(duration, hourly_rate_usd),
    )


def report_output_path(
    experiment_id: str,
    base: Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Default JSON output location for a given experiment's smoke report."""
    return base / f"smoke_{experiment_id}.json"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test orchestrator for Phase 18 / E1-E10 experiments.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=ALLOWED_EXPERIMENTS,
        help="Experiment ID from the registry.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Step count the eventual full run would do (informational).",
    )
    parser.add_argument(
        "--dry-run-after",
        action="store_true",
        help="Also confirm the experiment runner module + callable exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the report output path "
        "(default: reports/preflight/smoke_<id>.json).",
    )
    parser.add_argument(
        "--hourly-rate-usd",
        type=float,
        default=0.99,
        help="GPU hourly billing rate. Default: 0.99 (RTX 5090).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = run_smoke(
        args.experiment,
        max_steps=args.max_steps,
        dry_run_after=args.dry_run_after,
        hourly_rate_usd=args.hourly_rate_usd,
    )

    out_path = args.output or report_output_path(args.experiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))

    verdict = "PASS" if report.ok else "FAIL"
    print(
        f"[smoke] {args.experiment}: {verdict} "
        f"({report.duration_s:.1f}s, ~${report.estimated_cost_usd:.3f})"
    )
    for c in report.checks:
        marker = "[ok]" if c.ok else "[fail]"
        print(f"  {marker} {c.name}: {c.detail}")
    print(f"  report -> {out_path}")

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
