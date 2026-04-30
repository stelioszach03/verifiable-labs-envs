# Session log — vlabs-calibrate

Append-only timestamped log of work done on this package. Newest entry on top.

---

## 2026-04-30 — Phase 15.A, Phase 3 (Steps 4–8)

**Goal**: validate the calibrate API across five synthetic distributions,
ship three external demos, prove the wheel builds, and finish the README.

### Done
- Pre-flight: dropped exec bit on `PHASE_15A_PLAN.md` via
  `git update-index --chmod=-x` (commit `6e3bdb3`).
- Step 4: `tests/test_coverage_validation.py` — five parametrised cases
  (`gaussian`, `heavy_tail_t3`, `bimodal`, `sparse`,
  `structured_misspecified`) at `n_train=500`, `n_test=2000`, `α=0.1`.
  All pass at ±5pp (±7pp for the structured case). Empirical coverages:
  0.9150, 0.9035, 0.8960, 0.9015, 0.9120.
- Step 5: three demos under `examples/calibrate/`:
  - `01_humaneval_passfail.py` — binary, demonstrates both verified
    (q=0) and vacuous (q=1) regimes for binary nonconformity.
  - `02_math_exact_match.py` — exact-match judge with per-trace σ.
  - `03_gsm8k_step_validity.py` — continuous reward, ensemble σ.
  All three exit 0; eval coverage ≥ 0.886.
- Step 6: backwards-compat verification — repo baseline `458 passed,
  6 skipped` (unchanged), new package `61 passed`, ruff clean (after
  fixing two minor SIM/B905 hits in step-5 / step-4 code), restricted-
  path diff empty.
- Step 7: clean reinstall succeeds; `python -m build` produces
  `vlabs_calibrate-0.1.0a1-py3-none-any.whl` (18K) and
  `vlabs_calibrate-0.1.0a1.tar.gz` (27K). Build exit 0.
- Step 8: README updated with Coverage Validation results table and a
  Demos section linking the three example scripts.

### Tests
- `pytest packages/vlabs-calibrate/tests/` → 61 passed (test_calibrate
  23, test_core 16, test_coverage_validation 6, test_nonconformity 12,
  test_types 4).
- `pytest --no-header --ignore=tests/training --ignore=tests/test_reproducibility.py
   --ignore=tests/test_timeout.py --ignore=tests/fixtures` → 458 passed, 6 skipped.

### Open / future phases
- Phase 15.B: wire `packages/vlabs-calibrate/tests` into root `testpaths`
  + `.github/workflows/ci.yml`. Deferred to keep Phase 15.A strictly
  additive.
- 0.2.0: Mondrian / class-conditional conformal for binary tasks.

---

## 2026-04-30 — Phase 15.A, Phase 2 (Steps 1–3)

**Goal**: bootstrap the standalone package with the core conformal kernel
and the public `calibrate()` API; keep the 506-test repo-root suite green.

### Done
- Pulled origin/main (no-op, already at `302d6b0`).
- Created package skeleton under `packages/vlabs-calibrate/`:
  `pyproject.toml`, `README.md`, `CHANGELOG.md`, `LICENSE` (copy of repo
  root), `PHASE_15A_PLAN.md` (mirror of plan-mode artefact), `SESSION_LOG.md`,
  `src/vlabs_calibrate/{__init__.py,_version.py,core.py,nonconformity.py,
  types.py,calibrate.py}`, `tests/{__init__.py,conftest.py,test_core.py,
  test_calibrate.py,test_nonconformity.py,test_types.py}`,
  `examples/calibrate/README.md`.
- Step 2: copied conformal primitives from
  `src/verifiable_labs_envs/conformal.py` into
  `packages/vlabs-calibrate/src/vlabs_calibrate/core.py` (verbatim
  numerical behaviour, module docstring rewritten). Original is untouched.
- Step 3: implemented `vlabs_calibrate.calibrate` + `CalibratedRewardFn`
  + non-conformity registry + dataclass types.
- Incorporated three Phase-2 nits:
  1. `evaluate(tolerance=0.05)` parameter on `CalibratedRewardFn`.
  2. README quickstart self-contained (no undefined `past_runs`).
  3. `nonconformity="binary"` docstring caveats the degenerate case
     and points to Mondrian conformal as planned for 0.2.0.

### Tests
- `pytest -x` from repo root → 506 passing, 5 skipped (unchanged).
- `pytest -x packages/vlabs-calibrate/tests/` → all green.

### Open
- Phase 3 (Steps 4–8) pending Stelios approval: coverage validation
  suite, three external demos, full README, version + wheel build check.
- Tests-in-CI deferred to Phase 15.B (additive-only constraint for 15.A).
