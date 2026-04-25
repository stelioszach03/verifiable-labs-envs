#!/usr/bin/env python3
"""Validate a scaffolded Verifiable Labs env across four checks.

Runs four independent validations against the env directory at
``env_path``; each writes its result to the console and the script
exits non-zero if any check fails.

1. **Tests pass** — ``pytest <env_path>/tests``.
2. **Calibration coverage** — runs the env's ``run_baseline`` on
   ``--n-cal`` (default 50) fresh seeds, reads
   ``meta["coverage"]`` from each, asserts the mean is within
   ``--tolerance`` (default 0.05) of ``1 - alpha``.
3. **Procedural-regeneration count** — reads the env package's
   ``EFFECTIVE_INSTANCES`` constant and asserts ``> 1e15``.
4. **Adapter compatibility** — tries
   ``verifiers.load_environment(env_id)`` first, falls back to
   ``verifiable_labs_envs.load_environment``, then asserts
   ``env.generate_instance(0)`` succeeds.

Usage::

    python scripts/validate_env.py environments/seismic_fwi
    python scripts/validate_env.py environments/seismic_fwi --n-cal 200
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


class CheckFailure(RuntimeError):
    """Raised by a validation step when its assertion doesn't hold."""


def _print_header(idx: int, total: int, title: str) -> None:
    print(f"[{idx}/{total}] {title}")


def _resolve_env(env_path: Path) -> tuple[str, Any, Path]:
    """Locate the env's Python package and import it.

    Returns ``(env_id, package_module, package_dir)``.

    Strategy:
    - The directory passed in is expected to contain a single
      sub-directory whose name matches the package (snake_case env_py).
    - That sub-directory is added to ``sys.path`` so it can be imported
      as a top-level module.
    """
    if not env_path.is_dir():
        raise CheckFailure(f"env_path {env_path} is not a directory")

    candidates = [
        p for p in env_path.iterdir()
        if p.is_dir() and (p / "__init__.py").is_file()
    ]
    if len(candidates) != 1:
        raise CheckFailure(
            f"expected exactly one Python package directory under {env_path}; "
            f"found: {[p.name for p in candidates]}"
        )
    pkg_dir = candidates[0]
    sys.path.insert(0, str(env_path))
    try:
        module = importlib.import_module(pkg_dir.name)
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(f"importing {pkg_dir.name} failed: {exc}") from exc
    env_id = getattr(module, "ENV_ID", pkg_dir.name.replace("_", "-"))
    return env_id, module, pkg_dir


# ───────── checks ──────────────────────────────────────────────


def check_tests_pass(env_path: Path) -> None:
    tests_dir = env_path / "tests"
    if not tests_dir.is_dir():
        raise CheckFailure(f"missing tests/ under {env_path}")
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CheckFailure(
            "pytest failed:\n"
            + proc.stdout[-2000:]
            + "\n---stderr---\n"
            + proc.stderr[-2000:]
        )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
    print(f"    pytest: {summary}")


def check_calibration_coverage(
    module: Any,
    n_cal: int,
    tolerance: float,
) -> None:
    if not hasattr(module, "load_environment"):
        raise CheckFailure(
            f"{module.__name__} has no load_environment() — cannot calibrate"
        )
    env = module.load_environment(calibration_quantile=2.0, fast=True)
    coverages: list[float] = []
    target: float | None = None
    skipped: list[str] = []
    for seed in range(n_cal):
        try:
            scored = env.run_baseline(seed=seed)
        except NotImplementedError as exc:
            skipped.append(str(exc))
            break
        except Exception as exc:  # noqa: BLE001
            raise CheckFailure(
                f"run_baseline(seed={seed}) raised {type(exc).__name__}: {exc}"
            ) from exc
        meta = scored.get("meta", {})
        cov = meta.get("coverage")
        if cov is None:
            raise CheckFailure(
                "scored dict missing meta.coverage — your reward function "
                "must populate it (see reward.py / compute_reward)."
            )
        coverages.append(float(cov))
        if target is None:
            target = float(meta.get("target_coverage", 0.9))
    if skipped:
        raise CheckFailure(
            "calibration aborted: NotImplementedError raised in baseline "
            f"pipeline ({skipped[0]}). Fill the stubs and retry."
        )
    if not coverages:
        raise CheckFailure("no calibration samples — run_baseline produced none")
    mean_cov = sum(coverages) / len(coverages)
    delta = abs(mean_cov - (target or 0.9))
    print(
        f"    coverage: n={len(coverages)} mean={mean_cov:.4f} "
        f"target={target:.3f} |Δ|={delta:.4f} (tolerance ±{tolerance:.2f})"
    )
    if delta > tolerance:
        raise CheckFailure(
            f"calibration drift |Δ|={delta:.4f} exceeds ±{tolerance:.2f} "
            "tolerance — re-tune the conformal_quantile or check your "
            "non-conformity score."
        )


def check_procedural_regeneration(module: Any, threshold: float = 1e15) -> None:
    eff = getattr(module, "EFFECTIVE_INSTANCES", None)
    if eff is None:
        raise CheckFailure(
            f"{module.__name__} does not expose EFFECTIVE_INSTANCES "
            "(int constant in <pkg>/__init__.py). Set it to "
            "|seed_space| × |ground_truth_pool|."
        )
    eff_int = int(eff)
    print(f"    procedural: EFFECTIVE_INSTANCES = {eff_int:.3e}")
    if eff_int <= threshold:
        raise CheckFailure(
            f"EFFECTIVE_INSTANCES = {eff_int:.3e} is below the "
            f"{threshold:.0e} contamination-resistance threshold. "
            "Either grow your ground-truth pool or use a wider seed space."
        )


def check_adapter_compatibility(env_id: str, module: Any) -> None:
    """Try to load the env via the verifiers framework first; fall back
    to the in-tree ``verifiable_labs_envs.load_environment`` so the
    check works for both installed and pre-registered envs."""
    last_err: Exception | None = None
    for source_label, loader in [
        ("verifiers", _try_import_loader("verifiers")),
        ("verifiable_labs_envs", _try_import_loader("verifiable_labs_envs")),
    ]:
        if loader is None:
            continue
        try:
            env = loader(env_id)
        except (KeyError, ValueError, ModuleNotFoundError) as exc:
            last_err = exc
            continue
        try:
            inst = env.generate_instance(seed=0)
        except NotImplementedError as exc:
            raise CheckFailure(
                f"{source_label}.load_environment({env_id!r}) succeeded but "
                f"generate_instance raised NotImplementedError: {exc}"
            ) from exc
        if inst is None:
            raise CheckFailure(
                f"{source_label}.load_environment({env_id!r}).generate_instance "
                "returned None"
            )
        print(
            f"    adapter: {source_label}.load_environment({env_id!r}) → "
            f"{type(env).__name__} (ok via in-tree fallback)"
        )
        return
    # No loader resolved the env id.
    if hasattr(module, "load_environment"):
        # Last-ditch: call the package's own load_environment directly.
        try:
            env = module.load_environment(calibration_quantile=2.0)
            env.generate_instance(seed=0)
        except NotImplementedError as exc:
            raise CheckFailure(
                "package-local load_environment+generate_instance hit "
                f"NotImplementedError ({exc}). Fill the env stubs first."
            ) from exc
        print(
            "    adapter: verifiers/verifiable_labs_envs registries didn't "
            "know the env (not yet registered); the package-local "
            "load_environment works (ok)"
        )
        return
    raise CheckFailure(
        "no adapter could resolve the env id "
        f"{env_id!r}: {last_err}. Register it in "
        "src/verifiable_labs_envs/__init__.py:_REGISTRY or pass "
        "--skip-adapter-check."
    )


def _try_import_loader(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        return None
    mod = importlib.import_module(module_name)
    return getattr(mod, "load_environment", None)


# ───────── main ────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env_path", type=Path)
    parser.add_argument("--n-cal", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--skip-adapter-check", action="store_true")
    args = parser.parse_args()

    env_path: Path = args.env_path.resolve()
    print(f"validating: {env_path}")

    try:
        env_id, module, _pkg_dir = _resolve_env(env_path)
        print(f"    env_id: {env_id}")
    except CheckFailure as exc:
        print(f"FAIL: {exc}")
        return 1

    failures: list[tuple[str, str]] = []
    checks = [
        ("tests pass", lambda: check_tests_pass(env_path)),
        ("calibration coverage", lambda: check_calibration_coverage(
            module, args.n_cal, args.tolerance,
        )),
        ("procedural-regeneration > 1e15",
            lambda: check_procedural_regeneration(module)),
    ]
    if not args.skip_adapter_check:
        checks.append(
            ("adapter compatibility",
             lambda: check_adapter_compatibility(env_id, module))
        )
    total = len(checks)
    for i, (title, fn) in enumerate(checks, start=1):
        _print_header(i, total, title)
        try:
            fn()
        except CheckFailure as exc:
            print(f"    FAIL: {exc}")
            failures.append((title, str(exc)))

    print()
    if failures:
        print(f"❌ {len(failures)}/{total} checks failed:")
        for title, msg in failures:
            print(f"   - {title}: {msg.splitlines()[0]}")
        return 1
    print(f"✅ all {total} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
