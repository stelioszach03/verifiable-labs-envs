# Phase 15.A — `vlabs-calibrate` standalone package

> Plan deliverable. Plan-mode artifact at `~/.claude/plans/verifiable-labs-greedy-cocke.md`. Once approved, the same content is mirrored to `packages/vlabs-calibrate/PHASE_15A_PLAN.md` and committed to `main` (no Co-Authored-By trailer).

---

## 0. Pre-flight — state I observed

| Check | Observed | Implication |
|---|---|---|
| WSL repo HEAD | `302d6b0` | Differs from the `ad830385…` you cited. WSL is **behind** Colab. Plan: `git fetch && git pull` before Step 1. |
| `git status` clean? | **No** — `src/verifiable_labs_envs/cli.py` modified; untracked `examples/eval/`, `examples/training/`, `runs_local/`, `tests/test_reproducibility.py`, `tests/test_timeout.py`, `src/verifiable_labs_envs/repro.py`, etc. | These are M3/M4 work-in-progress from your Phase 13 run. **I will not stash, commit, or touch them.** Phase 15.A only adds files under `packages/vlabs-calibrate/`, so the dirty WT does not block this work. Flagging so you can decide. |
| Background processes | No `train_multienv` or `leaderboard` running on WSL | Clear to proceed. |
| `pytest` collection paths | Root `pyproject.toml` → `testpaths = ["tests", "packages/verifiable-labs/tests"]` | New tests in `packages/vlabs-calibrate/tests/` are **not** auto-discovered by `pytest` from repo root. See §6 risk register R1. |
| CI workflow | `.github/workflows/ci.yml` runs `pytest` (root only) + lints fixed paths only | New package is invisible to CI until its paths are added. **Out of scope for 15.A** per "ADDITIVE ONLY" rule; flagged for 15.B. |
| `.gitignore` | Covers `.env`, `*.key`, `__pycache__`, `.venv`, etc. | Sufficient. No edits needed. |
| Existing API surface I will reuse | `src/verifiable_labs_envs/conformal.py` (5 funcs), `src/verifiable_labs_envs/calibration.py` (`auto_calibrate`, `ConformalConfig`) | **Copy, do not move**. Source of truth for the math. Keeps 488+ env tests untouched. |

---

## 1. Executive summary

1. **Promote the existing conformal kernel** (`src/verifiable_labs_envs/conformal.py`) into a new standalone package `packages/vlabs-calibrate/` by *copying* — the original stays bit-for-bit identical so the 488 env tests keep passing.
2. **Generalize the calibration loop**: the existing `auto_calibrate` requires `Instance`/`Prediction` duck-typed objects. The new public API accepts plain `dict` traces and a user-supplied `reward_fn`, decoupling calibration from any specific env schema.
3. **Public API is one entry point**: `vlabs_calibrate.calibrate(reward_fn, traces, alpha=0.1)` returns a `CalibratedRewardFn` callable — the "5-line quickstart" shape you spec'd.
4. **Validate coverage on synthetic distributions** (Gaussian, heavy-tail t₃, bimodal, sparse, structured) — each must hit `(1−α) ± 5pp` empirical coverage on held-out test sets, proving the wrapper is correct before users touch it.
5. **Three external-only demos** (HumanEval-style, MATH, GSM8K-style) — none use VL envs. They prove the API works for arbitrary user reward fns, which is the funding-pitch claim.

Estimate: **23 hours net coding** spread over 5–7 calendar days. Phase 1 (this plan) = ~1 hour. Phase 2 (Steps 1–3) = ~9 hours. Phase 3 (Steps 4–8) = ~13 hours.

---

## 2. Architecture

```
packages/vlabs-calibrate/                   # NEW — sibling of packages/verifiable-labs/
├── pyproject.toml                          # standalone, hatchling, numpy-only deps
├── README.md                               # 5-line quickstart + API ref
├── CHANGELOG.md                            # 0.1.0a1 entry
├── LICENSE                                 # Apache-2.0 (copy from repo root)
├── PHASE_15A_PLAN.md                       # this plan (committed deliverable)
├── SESSION_LOG.md                          # append-only working log
├── src/
│   └── vlabs_calibrate/
│       ├── __init__.py                     # public re-exports + __version__
│       ├── core.py                         # COPY of conformal primitives (5 fns)
│       ├── calibrate.py                    # vc.calibrate() + CalibratedRewardFn
│       ├── nonconformity.py                # built-in score functions + registry
│       ├── types.py                        # Trace TypedDict + dataclasses
│       └── _version.py                     # single source of truth: 0.1.0a1
├── tests/
│   ├── __init__.py
│   ├── conftest.py                         # rng fixtures, synthetic generators
│   ├── test_core.py                        # parity vs verifiable_labs_envs.conformal
│   ├── test_calibrate.py                   # calibrate() + CalibratedRewardFn
│   ├── test_nonconformity.py               # each built-in score
│   ├── test_types.py                       # Trace coercion / validation
│   └── test_coverage_validation.py         # 5 distributions, (1-α)±5pp guarantee
└── examples/
    └── calibrate/
        ├── README.md
        ├── 01_humaneval_passfail.py        # binary reward → calibrated
        ├── 02_math_exact_match.py          # 0/1 with judge confidence as σ
        └── 03_gsm8k_step_validity.py       # continuous in [0,1] with σ from ensemble

DEPENDENCIES (single arrow = imports):
   tests/*           ──→  src/vlabs_calibrate/{__init__, core, calibrate, ...}
   src/calibrate.py  ──→  core, nonconformity, types
   src/nonconformity ──→  core
   src/types.py      ──→  (stdlib only)
   src/core.py       ──→  numpy

PUBLIC SURFACE (what `import vlabs_calibrate as vc` exposes):
   vc.calibrate         — entry point (function)
   vc.CalibratedRewardFn— callable wrapper class
   vc.CalibrationResult — per-call output dataclass
   vc.CoverageReport    — per-batch validation dataclass
   vc.Trace             — TypedDict (input shape spec)
   vc.core              — submodule with primitives
   vc.nonconformity     — submodule with score registry
   vc.__version__       — "0.1.0a1"
```

Zero runtime dependency on `verifiable_labs_envs`. Standalone install path:

```bash
pip install -e packages/vlabs-calibrate           # editable, dev
pip install vlabs-calibrate                       # from PyPI (post-publish)
```

---

## 3. Public API specification

### 3.1 `vlabs_calibrate.calibrate(...)`

```python
def calibrate(
    reward_fn: Callable[..., float],
    traces: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.1,
    nonconformity: str | Callable[[Mapping[str, Any], float], float] = "scaled_residual",
    eps: float = 1e-8,
    reward_kwargs_keys: Sequence[str] | None = None,
) -> CalibratedRewardFn:
    """Wrap `reward_fn` with split-conformal coverage guarantees.

    Parameters
    ----------
    reward_fn
        A callable returning a scalar reward. Signature is open: positional
        and/or keyword arguments. At calibration time we invoke it as
        ``reward_fn(**{k: trace[k] for k in reward_kwargs_keys})`` if
        ``reward_kwargs_keys`` is given, else as ``reward_fn(**trace_minus_meta)``
        where meta keys are {"reference_reward", "uncertainty", "_meta"}.
    traces
        Calibration set. Each trace MUST contain every key the chosen
        non-conformity score requires. For "scaled_residual" (default) that
        is ``"reference_reward"`` and ``"uncertainty"`` plus the args needed
        by ``reward_fn``. Minimum n=2; recommended n>=200 for stable quantile.
    alpha
        Nominal miscoverage in (0, 1). Default 0.1 → 90% target coverage.
    nonconformity
        Either a registry key (one of "scaled_residual", "abs_residual",
        "binary") or a custom callable ``(trace, predicted_reward) -> float``.
    eps
        Numerical floor for the σ denominator in "scaled_residual".
    reward_kwargs_keys
        Optional explicit list of trace keys to pass to ``reward_fn``.
        If None, all keys except {"reference_reward", "uncertainty", "_meta"}
        are forwarded. Use this when traces carry extra fields ``reward_fn``
        cannot accept.

    Returns
    -------
    CalibratedRewardFn
        A callable + dataclass-like object holding the calibrated quantile
        and a frozen reference to ``reward_fn``.

    Raises
    ------
    ValueError
        If ``alpha`` ∉ (0,1), if ``len(traces) < 2``, if a trace is missing
        required keys for the chosen non-conformity, or if any non-conformity
        score is non-finite.
    KeyError
        If ``reward_kwargs_keys`` references a key absent from a trace.
    TypeError
        If ``nonconformity`` is neither a known string nor a callable.
    """
```

### 3.2 `CalibratedRewardFn` (callable wrapper)

```python
@dataclass(frozen=True)
class CalibratedRewardFn:
    reward_fn: Callable[..., float]
    quantile: float                    # the (1−α) quantile of non-conformity scores
    alpha: float                       # nominal miscoverage
    n_calibration: int                 # number of traces used
    nonconformity_name: str            # e.g. "scaled_residual" or "<callable>"
    nonconformity_stats: Mapping[str, float]   # mean/std/median/max/min of scores
    reward_kwargs_keys: tuple[str, ...] | None

    def __call__(
        self,
        *args: Any,
        sigma: float | None = None,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Run reward_fn and return reward + conformal interval.

        Positional/keyword args are forwarded verbatim to reward_fn.
        ``sigma`` is the uncertainty for *this* call (only used when the
        non-conformity is scale-aware, e.g. "scaled_residual"). For
        scale-free scores ("abs_residual", "binary"), sigma is ignored.

        Returns
        -------
        CalibrationResult with .reward, .interval, .sigma, .quantile, .alpha,
        .target_coverage. .covered is None unless reference_reward is provided
        via the optional ``reference=`` kwarg.
        """

    def evaluate(self, traces: Sequence[Mapping[str, Any]]) -> CoverageReport:
        """Run on held-out traces; return empirical coverage + diagnostics.

        Each trace must contain ``reference_reward`` (and ``uncertainty`` for
        scale-aware scores) so we can check whether reference falls in the
        predicted interval.
        """
```

### 3.3 Result dataclasses

```python
@dataclass(frozen=True)
class CalibrationResult:
    reward: float
    interval: tuple[float, float]      # (lower, upper)
    sigma: float                       # σ used for the interval (0.0 for scale-free)
    quantile: float                    # passthrough from CalibratedRewardFn
    alpha: float
    target_coverage: float             # 1 - alpha
    covered: bool | None = None        # set only when caller provided reference

@dataclass(frozen=True)
class CoverageReport:
    target_coverage: float             # 1 - alpha
    empirical_coverage: float
    n: int
    n_in_interval: int
    interval_width_mean: float
    interval_width_median: float
    nonconformity: dict[str, float]    # mean, std, min, max, median
    quantile: float
    alpha: float
    passes: bool                       # |empirical - target| <= 0.05  (default tol)
```

### 3.4 `Trace` TypedDict

```python
class Trace(TypedDict, total=False):
    """Loose dict shape — actual required keys depend on the chosen nonconformity.

    For "scaled_residual": MUST include "reference_reward" and "uncertainty"
                            plus all kwargs reward_fn expects.
    For "abs_residual"   : MUST include "reference_reward".
    For "binary"         : MUST include "reference_reward" (treated as 0/1 label).
    """
    reference_reward: float
    uncertainty: float
    _meta: dict[str, Any]              # opaque user data, never forwarded to reward_fn
    # Plus arbitrary extra keys forwarded to reward_fn.
```

### 3.5 Built-in non-conformity scores (`vlabs_calibrate.nonconformity`)

| Name | Formula | When to use |
|---|---|---|
| `scaled_residual` | `|reward − reference| / max(σ, eps)` | continuous reward + per-sample σ available. Default. |
| `abs_residual` | `|reward − reference|` | continuous reward, no σ. Interval becomes `[r − q, r + q]`. |
| `binary` | `1.0 if reward != reference else 0.0` | 0/1 reward (HumanEval, MATH exact-match). Quantile becomes 0 or 1; interval at test time is `[reward, reward]` if score=0 (covered) else `[0, 1]` (vacuous). |

### 3.6 Error contract

| Condition | Exception |
|---|---|
| `alpha ∉ (0, 1)` | `ValueError` |
| `len(traces) < 2` | `ValueError` |
| Missing required trace keys | `ValueError` (with which keys + which trace index) |
| Unknown `nonconformity` string | `ValueError` (lists registered names) |
| `nonconformity` neither str nor callable | `TypeError` |
| Non-finite score from a trace | `ValueError` (with trace index + score) |
| Custom score callable raises | propagates with index annotation in `__cause__` |
| `reward_fn` raises during calibration | propagates with index annotation in `__cause__` |

---

## 4. File-by-file breakdown

| Path | New? | Purpose | LOC | Imports | Test plan |
|---|---|---|---|---|---|
| `packages/vlabs-calibrate/pyproject.toml` | NEW | hatchling build, name `vlabs-calibrate`, version `0.1.0a1`, dep `numpy>=1.26`, optional `[dev]` | ~70 | — | manual `pip install -e .` |
| `packages/vlabs-calibrate/README.md` | NEW | 5-line quickstart + API ref + coverage results link | ~120 | — | doctest: skipped (alpha) |
| `packages/vlabs-calibrate/CHANGELOG.md` | NEW | `0.1.0a1` first release entry | ~15 | — | — |
| `packages/vlabs-calibrate/LICENSE` | NEW | Apache-2.0 (verbatim copy from repo root LICENSE) | 202 | — | — |
| `packages/vlabs-calibrate/PHASE_15A_PLAN.md` | NEW | this plan, committed | ~600 | — | — |
| `packages/vlabs-calibrate/SESSION_LOG.md` | NEW | append-only timestamped log | grows | — | — |
| `packages/vlabs-calibrate/src/vlabs_calibrate/__init__.py` | NEW | `from .calibrate import calibrate, CalibratedRewardFn`; re-exports; `__version__` | ~30 | core, calibrate, types, nonconformity | covered by every test (import) |
| `packages/vlabs-calibrate/src/vlabs_calibrate/_version.py` | NEW | `__version__ = "0.1.0a1"` | 2 | — | — |
| `packages/vlabs-calibrate/src/vlabs_calibrate/core.py` | NEW (copy) | `split_conformal_quantile`, `scaled_residuals`, `interval`, `coverage`, `coverage_score` | ~95 | numpy | `test_core.py` parity + edge cases |
| `packages/vlabs-calibrate/src/vlabs_calibrate/nonconformity.py` | NEW | registry `_REGISTRY: dict[str, Callable]`; `scaled_residual`, `abs_residual`, `binary`; `register()` decorator | ~90 | core | `test_nonconformity.py` for each + custom callable |
| `packages/vlabs-calibrate/src/vlabs_calibrate/types.py` | NEW | `Trace` TypedDict, `CalibrationResult`, `CoverageReport` dataclasses | ~70 | typing, dataclasses | `test_types.py` |
| `packages/vlabs-calibrate/src/vlabs_calibrate/calibrate.py` | NEW | `calibrate()` function + `CalibratedRewardFn` class | ~180 | core, nonconformity, types | `test_calibrate.py` |
| `packages/vlabs-calibrate/tests/__init__.py` | NEW | empty | 0 | — | — |
| `packages/vlabs-calibrate/tests/conftest.py` | NEW | `rng` fixture (seed 42), synthetic generators | ~60 | numpy, pytest | — |
| `packages/vlabs-calibrate/tests/test_core.py` | NEW | parity vs `verifiable_labs_envs.conformal` (when installed) + standalone behaviour | ~120 | numpy, pytest | — |
| `packages/vlabs-calibrate/tests/test_calibrate.py` | NEW | API smoke, error paths, evaluate(), quantile monotonicity | ~180 | numpy, pytest | — |
| `packages/vlabs-calibrate/tests/test_nonconformity.py` | NEW | each built-in + custom callable + registry | ~80 | numpy, pytest | — |
| `packages/vlabs-calibrate/tests/test_types.py` | NEW | dataclass immutability, default values | ~30 | pytest | — |
| `packages/vlabs-calibrate/tests/test_coverage_validation.py` | NEW | 5-distribution coverage check, parametrised, n_train=500, n_test=2000, target ±5pp | ~150 | numpy, pytest | — |
| `packages/vlabs-calibrate/examples/calibrate/README.md` | NEW | overview + how to run | ~40 | — | — |
| `packages/vlabs-calibrate/examples/calibrate/01_humaneval_passfail.py` | NEW | mocked HumanEval-style binary reward, `nonconformity="binary"` | ~80 | numpy | manual run |
| `packages/vlabs-calibrate/examples/calibrate/02_math_exact_match.py` | NEW | exact-match with judge-confidence σ | ~100 | numpy | manual run |
| `packages/vlabs-calibrate/examples/calibrate/03_gsm8k_step_validity.py` | NEW | continuous step-validity reward, ensemble σ | ~120 | numpy | manual run |

**Total NEW code**: ~1,250 LOC + ~620 LOC tests + ~300 LOC examples = ~2,170 LOC.
**Total NEW files**: 24.
**Files modified in `src/`, `packages/verifiable-labs/`, root configs**: **0**.

---

## 5. Step-by-step execution order

### Step 1 — package skeleton (Phase 2 begins here)
**Files created** (10): `pyproject.toml`, `README.md` (stub), `CHANGELOG.md`, `LICENSE`, `src/vlabs_calibrate/__init__.py` (stub re-exporting `__version__` only), `src/vlabs_calibrate/_version.py`, `tests/__init__.py`, `tests/conftest.py` (stub), `examples/calibrate/README.md`, top-level `.gitignore` not needed (root one covers).
**Verification**: `pip install -e packages/vlabs-calibrate` succeeds. `python -c "import vlabs_calibrate; print(vlabs_calibrate.__version__)"` prints `0.1.0a1`.
**Backwards-compat check**: `pytest -x` from repo root → still 506 passing, 5 skipped.
**Rollback**: `rm -rf packages/vlabs-calibrate` + `pip uninstall vlabs-calibrate`.
**Time**: 2 h.

### Step 2 — copy conformal primitives into `core.py`
**Files created** (2): `src/vlabs_calibrate/core.py`, `tests/test_core.py`.
**Implementation**: copy the 5 functions from `src/verifiable_labs_envs/conformal.py` verbatim, rename module docstring to refer to vlabs-calibrate, keep behaviour identical. Tests assert byte-identical numeric outputs vs `verifiable_labs_envs.conformal` when both are installed (this is `pytest.importorskip("verifiable_labs_envs")` so the package is still standalone).
**Verification**: `pytest packages/vlabs-calibrate/tests/test_core.py -v` → all green. `pytest -x` from root → still 506 + 5.
**Rollback**: delete the 2 new files.
**Time**: 2 h.

### Step 3 — `calibrate()` + `CalibratedRewardFn` + types + nonconformity
**Files created** (5): `src/vlabs_calibrate/types.py`, `src/vlabs_calibrate/nonconformity.py`, `src/vlabs_calibrate/calibrate.py`, `tests/test_types.py`, `tests/test_nonconformity.py`, `tests/test_calibrate.py`. Update `__init__.py` to re-export.
**Verification**: `pytest packages/vlabs-calibrate/tests/ -v` → all green. Demo: open a Python REPL, run the 5-line quickstart with synthetic Gaussian data, confirm `result.interval` brackets `result.reward` and `evaluate()` returns ~90% empirical coverage.
**Rollback**: revert `__init__.py` and delete the 6 new files.
**Time**: 5 h.

> **Phase-2 gate**: stop here for your review. Commit a single feat commit with all of Step 1–3 (`feat(vlabs-calibrate): bootstrap package + conformal core + calibrate API`). Do NOT push until you say so.

### Step 4 — coverage validation suite
**Files created** (1): `tests/test_coverage_validation.py`.
**Implementation**: parametrised over 5 generators
  1. Gaussian: `r ~ Normal(predicted, σ_provided)`, σ_provided correct.
  2. Heavy-tail: `r ~ predicted + σ_provided · t(df=3)`.
  3. Bimodal: equal mixture of two Gaussians ±2σ around predicted.
  4. Sparse: `r = predicted` w.p. 0.7 else `r = predicted + σ_provided · N(0,1)·5`.
  5. Structured: `σ_provided` is mis-specified (off by 2× on half the calibration set) — tests robustness.
For each: `n_train=500`, `n_test=2000`, `α=0.1`. Assert `|empirical_coverage − 0.9| ≤ 0.05`. Distribution 5 is allowed `≤ 0.07` because mis-specified σ widens the legitimate band; flagged as expected.
**Verification**: `pytest tests/test_coverage_validation.py -v` (~30 s).
**Time**: 2 h.

### Step 5 — three external demos
**Files created** (3): `examples/calibrate/01_humaneval_passfail.py`, `examples/calibrate/02_math_exact_match.py`, `examples/calibrate/03_gsm8k_step_validity.py`. Each is fully self-contained — synthesises its own (prompt, completion, ground_truth, reference_reward, uncertainty) tuples in-script, no external dataset downloads. Adds `examples/calibrate/README.md` describing what each demo proves.
**Verification**: `python examples/calibrate/01_humaneval_passfail.py` etc., each prints a coverage table and exits 0.
**Time**: 5 h.

### Step 6 — backwards-compat hard verification (the critical step)
**Procedure** (run all four):
1. From repo root: `pytest -x` → expect ≥506 passing, 5 skipped (no regression).
2. `pytest -x packages/vlabs-calibrate/tests/` → expect all green (the new tests).
3. `ruff check src tests benchmarks examples scripts packages/verifiable-labs/src packages/verifiable-labs/tests` → unchanged from baseline.
4. `git diff --stat src/ packages/verifiable-labs/` → expect **empty diff**. Anything non-empty here halts the phase.
**Files modified**: 0 in restricted paths.
**Rollback**: if any check fails, `git restore` the offending file (only possible if I touched it accidentally, which the `git diff --stat` step exposes immediately).
**Time**: 2 h (mostly waiting for `pytest`).

### Step 7 — pip-install verification + version check
**Procedure**:
1. `pip uninstall -y vlabs-calibrate` (clean slate)
2. `pip install -e packages/vlabs-calibrate`
3. `python -c "import vlabs_calibrate as vc; print(vc.__version__); print(vc.calibrate)"`
4. `python -m build packages/vlabs-calibrate` → wheel + sdist build succeed.
**Verification**: wheel file present in `packages/vlabs-calibrate/dist/vlabs_calibrate-0.1.0a1-py3-none-any.whl`. (Not uploaded to PyPI in this phase — that's a separate user-driven action.)
**Time**: 1 h.

### Step 8 — README + API docs
**Files modified** (1, NEW): `packages/vlabs-calibrate/README.md`. Replace stub with full doc — quickstart, install, API reference (mirrors §3 of this plan), coverage validation results table from Step 4 output, link to demos.
**Verification**: visual diff. Any user reading README + running demo 01 should reach a working calibrated reward in < 2 minutes.
**Time**: 1 h.

> **Phase-3 gate**: stop here for your review. Single commit `feat(vlabs-calibrate): coverage validation, demos, full README` (or split per step if you prefer). Push to `main` only on your explicit go.

---

## 6. Risk register

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | Adding `packages/vlabs-calibrate/tests/` to root `testpaths` would edit an existing file, violating "ADDITIVE ONLY". So new tests are invisible to repo-root `pytest`. | Certain | Medium — tests don't run in CI | (a) Run new tests manually in local + Phase 2/3 gates. (b) Flag follow-up "Phase 15.B: extend testpaths + ci.yml" so tests join CI in a separate, clearly-scoped commit. (c) Document the manual command in `packages/vlabs-calibrate/README.md`. |
| **R2** | Hidden coupling: an env or `examples/training/` script imports from `verifiable_labs_envs.conformal` in a way that breaks if I edit it. | Low (I'm copying, not editing) | High | Step 6 step (4) `git diff --stat` enforces 0 lines changed in restricted paths. If I accidentally touch `conformal.py`, the gate fails before commit. |
| **R3** | Coverage validation fails for one of the 5 distributions (especially structured/mis-specified σ). | Medium | Medium | Plan tolerance per-distribution (5pp default, 7pp for the structured/mis-specified case). If a distribution legitimately violates, document it in the report rather than hide it — that's a real property of conformal under mis-specified σ. |
| **R4** | The user (Stelios) wants the API surface to look different after seeing this plan (e.g. different return type, different trace shape). | Medium | Low if caught now | This is exactly the point of the Phase-1 gate. We exit plan mode and you can request changes before any code is written. |
| **R5** | Working tree has uncommitted M3/M4 work. A `git pull` could conflict; a stray `git add .` in Step 1 could pull them into the calibrate commit. | Low (I won't `git add .`) | High (mixes phases) | Use only `git add packages/vlabs-calibrate/` for every commit. Run `git status` before each commit. Refuse to commit if anything outside `packages/vlabs-calibrate/` is staged. |

---

## 7. `pyproject.toml` draft

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "vlabs-calibrate"
version = "0.1.0a1"
description = "Conformal coverage guarantees for any reward function — wrap a Python callable with provable (1-α) coverage in 5 lines."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [
    { name = "Stelios Zacharioudakis", email = "sdi2200243@di.uoa.gr" },
]
keywords = [
    "conformal-prediction",
    "calibration",
    "reinforcement-learning",
    "rlvr",
    "reward-models",
    "uncertainty-quantification",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "build>=1.2",
]
scipy = [
    "scipy>=1.11",                # for advanced quantile methods, Wilcoxon, etc.
]

[project.urls]
Homepage = "https://github.com/stelioszach03/verifiable-labs-envs/tree/main/packages/vlabs-calibrate"
Source = "https://github.com/stelioszach03/verifiable-labs-envs/tree/main/packages/vlabs-calibrate"
Issues = "https://github.com/stelioszach03/verifiable-labs-envs/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/vlabs_calibrate"]

[tool.hatch.build.targets.sdist]
include = ["src/", "tests/", "examples/", "README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
```

---

## 8. `README.md` draft (first 50 lines)

```markdown
# vlabs-calibrate

> Conformal coverage guarantees for any reward function. Five lines of Python.

`vlabs-calibrate` wraps any Python reward callable with a split-conformal
prediction interval that provides marginal **(1 − α) coverage** under
exchangeability. Drop-in replacement for your reward function — get
calibrated intervals + a verified-coverage flag instead of a bare scalar.

The math is the [split-conformal procedure of Lei et al. (2018)](https://arxiv.org/abs/1604.04173).
The pitch: **every RL training run today ships uncalibrated rewards.**
`vlabs-calibrate` is the first step toward fixing that.

## Install

```bash
pip install vlabs-calibrate
```

Python `>=3.10`, single core dependency: `numpy`.

## Quickstart

```python
import vlabs_calibrate as vc

# Your reward function — could be anything, signature is open.
def my_reward(prompt, completion, ground_truth):
    return float(completion.strip() == ground_truth.strip())

# Calibration set: any iterable of dicts with the required keys.
traces = [
    {"prompt": p, "completion": c, "ground_truth": g,
     "reference_reward": r, "uncertainty": s}
    for (p, c, g, r, s) in past_runs
]

# Calibrate — one line.
calibrated = vc.calibrate(my_reward, traces, alpha=0.1)

# Use anywhere — drop-in replacement for `my_reward`.
result = calibrated("What is 2+2?", "4", "4", sigma=0.1)
print(result.reward, result.interval, result.target_coverage)
# → 1.0  (0.7, 1.0)  0.9
```
```

---

## 9. Sample test file — `tests/test_calibrate.py` (first ~80 LOC)

```python
"""Tests for vlabs_calibrate.calibrate() and CalibratedRewardFn."""
from __future__ import annotations

import numpy as np
import pytest

import vlabs_calibrate as vc
from vlabs_calibrate.types import CalibrationResult, CoverageReport


def _gaussian_traces(n: int, *, seed: int) -> list[dict]:
    """Synthetic traces: reference ~ predicted + N(0, sigma)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = float(rng.standard_normal())
        sigma = 0.5
        ref = x + sigma * float(rng.standard_normal())
        out.append({"x": x, "reference_reward": ref, "uncertainty": sigma})
    return out


def _identity_reward(*, x: float) -> float:
    return float(x)


def test_calibrate_basic_returns_callable():
    traces = _gaussian_traces(200, seed=0)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    assert callable(cal)
    assert isinstance(cal, vc.CalibratedRewardFn)
    assert 0 < cal.quantile < 10
    assert cal.alpha == 0.1
    assert cal.n_calibration == 200


def test_calibrate_call_returns_result():
    traces = _gaussian_traces(200, seed=1)
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1)
    result = cal(x=1.0, sigma=0.5)
    assert isinstance(result, CalibrationResult)
    assert result.reward == pytest.approx(1.0)
    lo, hi = result.interval
    assert lo < result.reward < hi
    assert result.target_coverage == pytest.approx(0.9)


def test_calibrate_evaluate_hits_target_coverage():
    cal = vc.calibrate(_identity_reward, _gaussian_traces(500, seed=2), alpha=0.1)
    held_out = _gaussian_traces(2000, seed=3)
    report = cal.evaluate(held_out)
    assert isinstance(report, CoverageReport)
    assert abs(report.empirical_coverage - 0.9) < 0.05
    assert report.passes


def test_calibrate_rejects_bad_alpha():
    traces = _gaussian_traces(20, seed=4)
    with pytest.raises(ValueError, match="alpha"):
        vc.calibrate(_identity_reward, traces, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        vc.calibrate(_identity_reward, traces, alpha=1.0)


def test_calibrate_rejects_tiny_calibration_set():
    with pytest.raises(ValueError, match="at least 2"):
        vc.calibrate(_identity_reward, [_gaussian_traces(1, seed=5)[0]], alpha=0.1)


def test_calibrate_rejects_missing_keys():
    bad_traces = [{"x": 1.0}]  # missing reference_reward + uncertainty
    with pytest.raises(ValueError, match="reference_reward"):
        vc.calibrate(_identity_reward, bad_traces * 5, alpha=0.1)


def test_calibrate_custom_nonconformity_callable():
    traces = _gaussian_traces(200, seed=6)
    def custom(trace, predicted):
        return abs(trace["reference_reward"] - predicted)  # abs_residual by hand
    cal = vc.calibrate(_identity_reward, traces, alpha=0.1, nonconformity=custom)
    assert cal.nonconformity_name == "<callable>"


def test_calibrate_quantile_increases_when_alpha_decreases():
    traces = _gaussian_traces(500, seed=7)
    cal_loose = vc.calibrate(_identity_reward, traces, alpha=0.20)
    cal_tight = vc.calibrate(_identity_reward, traces, alpha=0.05)
    assert cal_tight.quantile > cal_loose.quantile
```

---

## 10. Estimated completion

| Step | Task | Hours | This session? |
|---|---|---|---|
| 1 | package skeleton + LICENSE + stub README + pyproject | 2 | ✅ yes (Phase 2) |
| 2 | `core.py` + `test_core.py` (copy of conformal primitives) | 2 | ✅ yes (Phase 2) |
| 3 | `types.py` + `nonconformity.py` + `calibrate.py` + 3 test files | 5 | ✅ yes (Phase 2) |
| **Phase-2 gate** | commit + your review | — | ⏸ stop here |
| 4 | coverage validation suite (5 distributions) | 2 | ⏭ Phase 3 |
| 5 | three external demos | 5 | ⏭ Phase 3 |
| 6 | backwards-compat verification (run pytest, ruff, diff) | 2 | ⏭ Phase 3 |
| 7 | pip-install + wheel build verification | 1 | ⏭ Phase 3 |
| 8 | full README + CHANGELOG | 1 | ⏭ Phase 3 |
| **Phase-3 gate** | final commit + push on your go | — | ⏸ stop here |
| **Total** | | **20 h** | |

Realistically: **Phase 2 fits in this session if you approve quickly** (9 h of focused work, but I can run Steps 1–3 in parallel-ish). **Phase 3** is separate session(s).

---

## 11. Open questions for Stelios (please answer before I exit plan mode)

1. **Working-tree state**: WSL HEAD is `302d6b0`, you cited `ad830385`. Do you want me to `git fetch && git pull origin main` *before* Step 1 to sync, or work off the current WSL state? (Recommend: pull first, since main moved on the Colab VM and we'll push from WSL.)
2. **Trace dict key conventions**: I'm using `reference_reward` (the gold/label) and `uncertainty` (σ). Are those names OK, or do you want `gold_reward` / `sigma`? Funded-startup-pitch-readability matters here.
3. **`nonconformity` callable signature**: my draft is `f(trace, predicted_reward) -> float`. Alternative: `f(trace) -> float` (trace contains both `reward` and `reference_reward`). The first is cleaner because `predicted_reward` is recomputed at calibration time. OK?
4. **Demo dataset stance**: I plan to **synthesise** all demo data inline (no HumanEval/MATH/GSM8K downloads) so the examples run with `python examples/calibrate/01_humaneval_passfail.py` zero-config. Acceptable, or do you want real-dataset wiring?
5. **Tests-in-CI**: per R1, new tests aren't picked up by root `pytest` without editing root `pyproject.toml`. Stay strict ("ADDITIVE ONLY", new tests run separately, follow-up phase wires CI) or relax for this single line?

Once you answer these (or wave them off), I exit plan mode and proceed to Phase 2 = Steps 1–3.
