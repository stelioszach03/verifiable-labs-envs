"""Tests for ``scripts/validate_env.py``.

Two flavours:
- Scaffold-state runs (NotImplementedError stubs): tests + procedural
  pass; calibration + adapter fail. Validator exits non-zero with the
  right error messages.
- Filled-in runs (a tiny but complete env we render in tmp_path): all
  four checks pass.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CREATE = REPO_ROOT / "scripts" / "create_env.py"
VALIDATE = REPO_ROOT / "scripts" / "validate_env.py"


def _scaffold(target: Path, env_id: str = "demo-env") -> None:
    proc = subprocess.run(
        [sys.executable, str(CREATE), env_id, "--domain", "demo",
         "--target", str(target)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def _validate(target: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATE), str(target), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def _patch_to_minimal_working_env(target: Path, env_py: str) -> None:
    """Replace the NotImplementedError stubs with a simplest-possible
    working impl: scalar identity problem ``y = x + noise``."""
    pkg = target / env_py
    (pkg / "data.py").write_text(_MINIMAL_DATA)
    (pkg / "forward_op.py").write_text(_MINIMAL_FORWARD)
    # env.py + reward.py + adapter.py from the scaffold are already
    # functional once data + forward are filled.
    # baseline_predict already returns zeros, which matches what we
    # need for the calibration to produce a meaningful coverage rate
    # against a unit-noise gaussian.


# ── scaffold-state validation ──────────────────────────────────


def test_validate_unfilled_scaffold_fails_calibration_and_adapter(tmp_path):
    target = tmp_path / "scaffold"
    _scaffold(target, "demo-env")
    proc = _validate(target)
    assert proc.returncode != 0
    out = proc.stdout
    # Tests pass even on stub-state (NotImplementedError tests skip).
    assert "[1/4] tests pass" in out
    assert "passed" in out and "skipped" in out
    # Procedural-regeneration also passes (just reads the constant).
    assert "EFFECTIVE_INSTANCES" in out
    # Calibration aborts with NotImplementedError.
    assert "calibration aborted" in out
    # Adapter check fails because generate_instance crashes.
    assert "NotImplementedError" in out


def test_validate_skip_adapter_check(tmp_path):
    target = tmp_path / "scaffold-skip"
    _scaffold(target, "demo-env")
    proc = _validate(target, "--skip-adapter-check")
    out = proc.stdout
    assert "[3/3]" in out  # only 3 checks now
    # The validator's own check headers ([N/4]) must not mention adapter;
    # we only inspect lines starting with "[" so the tmp_path directory
    # name doesn't trigger a false positive.
    headers = [ln for ln in out.splitlines() if ln.lstrip().startswith("[")]
    assert not any("adapter" in h for h in headers)


def test_validate_complains_when_env_path_not_a_directory(tmp_path):
    f = tmp_path / "not-a-dir.txt"
    f.write_text("oops")
    proc = _validate(f)
    assert proc.returncode != 0
    assert "is not a directory" in proc.stdout


def test_validate_complains_when_no_python_package(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _validate(empty)
    assert proc.returncode != 0
    assert "Python package" in proc.stdout or "Python package" in proc.stderr


# ── filled-in env validation ───────────────────────────────────


def test_validate_filled_env_passes_all_four_checks(tmp_path):
    target = tmp_path / "filled"
    _scaffold(target, "demo-env")
    _patch_to_minimal_working_env(target, "demo_env")
    proc = _validate(target, "--n-cal", "30", "--tolerance", "0.10")
    assert proc.returncode == 0, proc.stdout + "\n---stderr---\n" + proc.stderr
    out = proc.stdout
    assert "[1/4] tests pass" in out
    assert "[2/4] calibration coverage" in out
    assert "[3/4] procedural-regeneration > 1e15" in out
    assert "[4/4] adapter compatibility" in out
    assert "all 4 checks passed" in out


# ── minimal working env source (replaces NotImplementedError stubs) ─


_MINIMAL_DATA = '''\
"""Minimal generator: ground truth = uniform noise, public input = noisy y."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Instance:
    y: np.ndarray
    x_true: np.ndarray
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_inputs(self) -> dict[str, Any]:
        return {"y": self.y, **self.metadata}


@dataclass(frozen=True)
class Prediction:
    x_hat: np.ndarray
    sigma_hat: np.ndarray


def generate_ground_truth(seed: int, **hyperparams: Any) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(8,))
'''

_MINIMAL_FORWARD = '''\
"""Identity forward operator with no extra state."""
from __future__ import annotations

import numpy as np


def forward(x: np.ndarray, *, seed: int = 0) -> np.ndarray:
    return x.copy()


def adjoint(y: np.ndarray) -> np.ndarray:
    return y.copy()
'''
