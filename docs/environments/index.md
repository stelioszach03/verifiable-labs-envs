# Environments

The v0.1 platform ships **25 environments** spanning compressed
sensing, computational imaging, physics inversion, rendering,
symbolic algebra, code execution, tool orchestration, text-to-SQL,
and long-context retrieval / synthesis / reasoning. Phase 27 closes
the env-catalogue track at 25 envs across 7 template families;
subsequent work prioritises customer-validated demand.

| env id | domain | turns | tools | status |
|---|---|---|---|---|
| [`sparse-fourier-recovery`](sparse-fourier-recovery.md) | compressed-sensing | 1 | no | ✅ stable |
| `sparse-fourier-recovery-multiturn` | compressed-sensing | 3 | no | ✅ stable |
| `sparse-fourier-recovery-tools` | compressed-sensing | 3 | yes | ✅ stable |
| [`lodopab-ct-simplified`](lodopab-ct.md) | imaging-CT | 1 | no | ✅ stable |
| `lodopab-ct-simplified-multiturn` | imaging-CT | 3 | no | ✅ stable |
| [`mri-knee-reconstruction`](mri-knee.md) | imaging-MRI | 1 | no | ✅ stable |
| `mri-knee-reconstruction-multiturn` | imaging-MRI | 3 | no | ✅ stable |
| [`phase-retrieval`](phase-retrieval.md) | physics-inverse | 1 | no | ✅ stable |
| `phase-retrieval-tools` | physics-inverse | 3 | yes | ✅ stable |
| [`super-resolution`](super-resolution.md) | imaging-SR | 1 | no | ✅ stable |
| `math-algebra` | symbolic-algebra | 1 | no | 🆕 alpha (Phase 21) |
| `math-algebra-multiturn` | symbolic-algebra | 3 | no | 🆕 alpha (Phase 21) |
| `math-algebra-tools` | symbolic-algebra | 3 | yes | 🆕 alpha (Phase 21) |
| [`code-humaneval`](code-humaneval.md) | code-execution | 1 | no | 🆕 alpha (Phase 24) |
| [`code-humaneval-multiturn`](code-humaneval-multiturn.md) | code-execution | 3 | no | 🆕 alpha (Phase 24) |
| [`code-humaneval-tools`](code-humaneval-tools.md) | code-execution | 3 | yes | 🆕 alpha (Phase 24) |
| [`code-mini-repo`](code-mini-repo.md) | code-execution-repo | 1 | no | 🆕 alpha (Phase 24) |
| [`tool-calling-single`](tool-calling-single.md) | tool-orchestration | 1 | yes | 🆕 alpha (Phase 25) |
| [`tool-calling-multiturn`](tool-calling-multiturn.md) | tool-orchestration | N | yes | 🆕 alpha (Phase 25) |
| [`tool-calling-debug`](tool-calling-debug.md) | tool-orchestration-debug | 1 | yes | 🆕 alpha (Phase 25) |
| [`sql-single-turn`](sql-single-turn.md) | text-to-SQL | 1 | no | 🆕 alpha (Phase 26) |
| [`sql-multiturn`](sql-multiturn.md) | text-to-SQL | 3 | no | 🆕 alpha (Phase 26) |
| [`long-context-needle`](long-context-needle.md) | long-context-retrieval | 1 | no | 🆕 alpha (Phase 27) |
| [`long-context-synthesis`](long-context-synthesis.md) | long-context-synthesis | 3 | no | 🆕 alpha (Phase 27) |
| [`long-context-reasoning`](long-context-reasoning.md) | long-context-reasoning | 1 | no | 🆕 alpha (Phase 27) |

Six env families are documented in detail in this section; the
multi-turn and tool-use variants share the underlying problem and
reward and only differ in the dispatch shape (see
[Multi-turn](../concepts/multi-turn.md) and
[Tool use](../concepts/tool-use.md)). The Phase 21 `math-algebra`
family is the first non-inverse-problem domain — it ships under the
new `templates/symbolic-math/` family and uses SymPy-verified
equivalence (`simplify(answer − gold) == 0`) instead of a numeric
NMSE / σ̂ scoring kernel.

## Common interface

Every env exposes:

```python
env.generate_instance(seed: int) -> Instance     # bit-deterministic
env.score(prediction, instance) -> dict          # reward + components
env.run_baseline(seed: int) -> dict              # classical solver baseline
env.run_rollout(solver, instance) -> dict        # multi-turn (where supported)
```

Plus an `EnvAdapter` for LLM dispatch:

```python
adapter.system_prompt: str
adapter.build_user_prompt(instance) -> str
adapter.parse_response(text, instance) -> Prediction
```

## Reward shape

All envs produce a normalised `reward ∈ [0, 1]` plus a `components`
dict so you can attribute the score:

```python
{
  "reward": 0.554,
  "components": {
    "nmse": 0.62,        # point-estimate fidelity
    "support": 0.85,     # support-recovery fraction (where applicable)
    "conformal": 0.76,   # uncertainty-coverage indicator
  },
  "coverage": 0.91,      # empirical conformal coverage on this seed pool
  "parse_ok": True,
  "complete": True,
  "meta": {...}          # env-specific extras
}
```

The exact weighting is published per env in the env's source and the
paper's methodology section.

## Adding a new env

See [Tutorials → Creating a custom env](../tutorials/creating-custom-env.md).
The scaffold at `templates/inverse-problem/` covers the most common
shape (1D / 2D inverse problem with a closed-form forward operator
and a Gaussian noise model).
