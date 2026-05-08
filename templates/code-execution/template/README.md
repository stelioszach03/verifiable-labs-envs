# __ENV_ID__

Verifiable Labs code-execution RL environment for **__DOMAIN__**.

This package was scaffolded from
``templates/code-execution/`` via
``scripts/create_env.py __ENV_ID__ --template code-execution --domain "__DOMAIN__"``.

## What this env does

The env hands the solver a **function signature + docstring + visible
test cases**, expects a Python source string implementing the function,
and runs that source against a **hidden test suite** in a sandboxed
subprocess. Scoring is graded — the solver gets credit for every
hidden test that passes, not just an all-or-nothing pass/fail. This
is the continuous signal that conformal calibration needs.

| Component       | Weight | What it rewards                                          |
|-----------------|--------|----------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing a `code` field       |
| `parse_valid`   | 0.20   | Extracted code compiles via `compile(..., "exec")`       |
| `pass_rate`     | 0.70   | Fraction of (visible ∪ hidden) pytest cases that passed  |

The pytest invocation runs inside the sandbox primitive shipped at
``verifiable_labs_envs.sandbox.execute_in_sandbox_sync``. D5 limits
locked in PHASE_24_PLAN.md:

- 512 MB virtual address space cap (RLIMIT_AS)
- 30 s wall-clock + 20 s CPU-seconds (RLIMIT_CPU)
- ``unshare -rn`` network isolation (Linux only — outbound socket
  connect fails, defending the host against arbitrary egress)
- 16-process fanout cap (RLIMIT_NPROC)
- 64 MB max-created-file (RLIMIT_FSIZE)
- Tmpfs-backed scratch dir wiped on every exit path

## Filling in the scaffold

Replace the `NotImplementedError` stubs in:

- ``__ENV_PY__/data.py`` — `generate_problem(seed, **hyperparams)`
  returns a dict with `function_signature`, `docstring`,
  `visible_tests`, `hidden_tests`, `gold_solution`, `template_name`.
  Use `numpy.random.default_rng(seed)` for reproducibility; ensure the
  seed × parameter-range product gives `EFFECTIVE_INSTANCES > 1e15`.
- ``__ENV_PY__/env.py`` — adjust hyperparams (sandbox timeout, alpha,
  weights) if the domain needs tighter or looser bounds.

The scoring kernel and adapter need no edits in most cases; the
default JSON envelope (``{"code": "...", "confidence": <float>}``)
and pytest-pass-rate signal cover the common case.

## Why a separate template family

The inverse-problem template hard-codes forward operators, NMSE
scoring, and per-entry σ̂ vectors. The symbolic-math template
hard-codes SymPy-string equivalence. Neither applies to code
execution: the verification primitive is **subprocess-bounded test
running**, not in-process algebra. This family swaps in:

- A `Problem` shape carrying separated `visible_tests` (shown to the
  model in the prompt) and `hidden_tests` (oracle, never leaked).
- A reward kernel that delegates the heavy lifting to
  ``verifiable_labs_envs.sandbox.execute_in_sandbox_sync``.
- A per-call tmpdir lifecycle that cleans up on every exit path.

## Running

```bash
python scripts/validate_env.py environments/__ENV_PY__/   # contract checks
pytest                                                     # unit + reward + sandbox tests
```

## Trusted-input scope

This env's sandbox runs under the D2-A "subprocess + rlimit" mechanism
(PHASE_24_PLAN.md §5). The locked guarantee is *isolation between
concurrent customer calls on the same machine*, not *defence against a
determined attacker who has compromised an API key*. A future public
"submit code anonymously" surface would require flipping the sandbox
to D2-B (Docker) or D2-C (Firecracker); the upgrade-gate sentinel test
in `tests/test_sandbox.py` keeps that gate visible.
