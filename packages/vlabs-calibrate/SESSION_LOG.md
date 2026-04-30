# Session log — vlabs-calibrate

Append-only timestamped log of work done on this package. Newest entry on top.

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
