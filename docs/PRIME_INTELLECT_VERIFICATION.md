# Prime Intellect Hub — fresh-venv install verification

Sprint 1 Polish Task A deliverable. Every one of the six Hub envs has been
round-trip verified from a clean Python 3.11 virtualenv against the
`stelioszach/<env-id>` Hub artifact, on 2026-04-24.

## TL;DR

| env id | Hub version | install rc | `load_environment` | baseline (seed=0) |
|---|---|---|---|---|
| `stelioszach/sparse-fourier-recovery` | 0.2.0 | 0 | OK | reward=0.9309 |
| `stelioszach/sparse-fourier-recovery-multiturn` | 0.2.0 | 0 | OK | reward=0.9309 |
| `stelioszach/sparse-fourier-recovery-tools` | 0.2.0 | 0 | OK | reward=0.9309 |
| `stelioszach/super-resolution-div2k-x4` | 0.2.0 | 0 | OK | n/a (slow, needs DIV2K) |
| `stelioszach/lodopab-ct-simplified` | 0.2.0 | 0 | OK | n/a (slow, needs phantom) |
| `stelioszach/lodopab-ct-simplified-multiturn` | 0.2.0 | 0 | OK | n/a (slow, needs phantom) |

6/6 pass. Reproducer below.

## What was broken in v0.1.0

Two artifacts of the Sprint-1 Phase-5 push only surfaced when we actually
installed the Hub artifacts from scratch:

1. **`verifiers>=0.1.13` pin artifact.** At push time the `0.1.13` line was
   dev-only on PyPI. PEP 440 treats `0.1.13.devN` as `< 0.1.13`, so
   `pip install` rejected every dev build even with `--pre`. Consumers got:

   ```
   ERROR: Could not find a version that satisfies the requirement
   verifiers>=0.1.13 (from <env-id>) (from versions: ..., 0.1.12,
   0.1.13.dev1, ..., 0.1.13.dev7)
   ```

   **Corrected** in
   [`scripts/populate_prime_envs.py`](../scripts/populate_prime_envs.py)
   (pin widened to `verifiers>=0.1.12`) and the six
   [`environments/*/pyproject.toml`](../environments/) files regenerated.

2. **Missing `env_id` / `env_args` on env instances.**
   `verifiers.load_environment` does
   `env_instance.env_id = env_instance.env_id or env_id` on the returned
   object. Our `SparseFourierEnv`, `LodopabCtEnv`, and `SuperResolutionEnv`
   classes did not declare those attributes, so every consumer hit
   `AttributeError: ... object has no attribute 'env_id'`.

   **Corrected** in
   [`src/verifiable_labs_envs/envs/sparse_fourier.py`](../src/verifiable_labs_envs/envs/sparse_fourier.py),
   [`lodopab_ct.py`](../src/verifiable_labs_envs/envs/lodopab_ct.py), and
   [`super_resolution.py`](../src/verifiable_labs_envs/envs/super_resolution.py)
   by adding `self.env_id = ""` + `self.env_args = {}` to each `__init__`.

Both fixes landed in monorepo commit `b901479`, then the six Hub envs were
re-pushed as v0.2.0. The table above is the post-fix verification.

## Reproducer

Prerequisites: Python 3.11 on PATH, `prime login` already completed.

```bash
# 1. Clean venv with only prime + verifiers.
VENV=/tmp/prime-verify-$(date +%Y%m%d-%H%M%S)
python3.11 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install prime verifiers

# 2. Pull each env (they are currently private) and install.
PULL_DIR=/tmp/prime-verify-pulls
mkdir -p "$PULL_DIR" && cd "$PULL_DIR"
for env in sparse-fourier-recovery \
           sparse-fourier-recovery-multiturn \
           sparse-fourier-recovery-tools \
           super-resolution-div2k-x4 \
           lodopab-ct-simplified \
           lodopab-ct-simplified-multiturn; do
    "$VENV/bin/prime" --plain env pull "stelioszach/$env" -t "$env"
    (cd "$env" && "$VENV/bin/pip" install -e .)
done

# 3. Round-trip check.
"$VENV/bin/python" <<'PY'
from verifiers import load_environment
for env_id in [
    "sparse-fourier-recovery",
    "sparse-fourier-recovery-multiturn",
    "sparse-fourier-recovery-tools",
    "super-resolution-div2k-x4",
    "lodopab-ct-simplified",
    "lodopab-ct-simplified-multiturn",
]:
    e = load_environment(env_id)
    print(env_id, "=>", e.name, "env_id:", e.env_id)
PY
```

Expected output: six lines, each matching `<env_id> => <env_id> env_id: <env_id>`.
For the three sparse-Fourier variants you can additionally call
`e.run_baseline(seed=0)["reward"]` and get `0.9308768…` (classical OMP baseline
on instance seed=0).

## Why the dev `verifiers` works now

With the widened pin `verifiers>=0.1.12`, pip picks the latest
`0.1.12` stable by default. If the consumer wants the dev line
(currently `0.1.13.dev7`), `--pre` opts in. Both paths satisfy the Hub
wrapper's requirement.

The `env_id` fix is orthogonal: regardless of which verifiers version
the consumer has, our env instances now carry the attribute the base
`load_environment` helper expects.

## What this does NOT prove

- Real-data slow paths (LoDoPaB real CT slices, DIV2K super-resolution,
  multi-turn LLM rollouts) are not exercised here. Those require either the
  `ct-real` extras or an `OPENROUTER_API_KEY`, which is out of scope for a
  pure install-verification run. They are exercised by the repo's pytest
  suite (184 green on 2026-04-24 after Task B's primitive tests) against
  the local checkout.
- The `prime env install stelioszach/<env>` one-shot install is blocked for
  private envs (the CLI instructs users to pull + `pip install -e .` instead,
  which is what the reproducer above does). Public visibility is a separate
  post-YC-submission task.

## Post-Task-B recheck — tools env v0.3.0 from Hub

After Task B pushed the primitive-composition redesign of
`sparse-fourier-recovery-tools` as v0.3.0 to the Hub, the Task-A fresh-venv
reproducer was re-run just for this env:

```bash
"$VENV/bin/prime" --plain env pull stelioszach/sparse-fourier-recovery-tools -t tools
(cd tools && "$VENV/bin/pip" install -e . --no-cache-dir)
"$VENV/bin/python" -c "
from verifiers import load_environment
from verifiable_labs_envs.envs.sparse_fourier_tools import TOOL_SCHEMAS
e = load_environment('sparse-fourier-recovery-tools')
tool_names = sorted(t['function']['name'] for t in TOOL_SCHEMAS)
print('tools:', tool_names)
print('baseline:', e.run_baseline(seed=0)['reward'])
"
```

Observed:

```
tools: ['compute_residual_tool', 'fft_tool', 'ifft_tool', 'sparsity_norm_tool', 'threshold_tool']
baseline: 0.9308768169392316
```

5/5 expected primitives present, `ista_tool` absent, baseline reward
matches the single-turn env as expected (same underlying instance +
classical OMP). **One caveat for external reproducers**: on the first
install after a `@main` push, pip's cache can serve the previous Git-dep
checkout; pass `--no-cache-dir` (as above) or `pip install ... --force-reinstall`
to guarantee the latest monorepo source is pulled.
