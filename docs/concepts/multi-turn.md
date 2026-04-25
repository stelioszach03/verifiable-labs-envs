# Multi-turn dialogue

Most v0.1 envs are single-turn (one prompt, one answer, one reward).
Three are multi-turn — the model gets feedback after each attempt and
can revise.

## Multi-turn envs

| env | turns | feedback type |
|---|---|---|
| `sparse-fourier-recovery-multiturn` | up to 3 | Fourier-domain residual `r = y - A(x_hat)` of the previous answer |
| `lodopab-ct-simplified-multiturn` | up to 3 | FBP-domain residual image of the previous reconstruction |
| `mri-knee-reconstruction-multiturn` | up to 3 | residual k-space data + a small recon hint |

## What "multi-turn" means in the platform

Each turn the env sends a fresh user message containing **structured
feedback** about the previous prediction. The model emits another
JSON answer in the same schema. The reward at every turn is computed
against the latest prediction; the rollout halts when the model
either:

- emits the same answer twice in a row (no progress), or
- fails to parse on a turn after at least one successful turn (the
  last good prediction is scored), or
- hits `max_turns` (default 3).

If the very first turn fails to parse, the rollout raises
`LLMSolverError` — we don't fabricate partial credit for a model that
can't follow the documented schema even once.

## Reward semantics across turns

The platform records per-turn rewards in `meta.turn_rewards` so you
can ask:

- *did the model improve from turn 1 to turn 3?*
- *did it overshoot* — got a good answer on turn 1, then drifted?
- *was the residual feedback informative?* (single-turn baseline vs.
  multi-turn: if the multi-turn mean is no higher, the model isn't
  using the feedback)

The benchmark in `paper/` shows that multi-turn helps on sparse-Fourier
(+0.04 over single-turn for Claude Haiku 4.5) but is roughly neutral
on CT and MRI (residual feedback in image space is harder to use
from an LLM's perspective than residual feedback in Fourier space).

## The dispatch layer

In v0.1 the **API** records every submission for a multi-turn env but
does **not** automatically replay them through `env.run_rollout`.
The session-state response includes the turn history; the SDK assembles
follow-up prompts client-side from that history.

This is a deliberate v0.1 simplification: it lets the API stay
stateless apart from session scope, and the SDK does the orchestration
the env's `run_rollout` would have done in-process. Tier 2 (v0.2) will
move turn dispatch server-side.

## Implementation pointer

If you're writing a multi-turn env from scratch, the canonical
reference is
`src/verifiable_labs_envs/envs/sparse_fourier_multiturn.py::SparseFourierMultiturnEnv`.
The minimum surface to implement is:

- `class FooMultiturnEnv(FooEnv)` — extend the single-turn env
- `def run_rollout(solver, instance, *, max_turns=...) -> ScoredResult`
  — loop over `complete_turns(messages)`, parse, score, build the next
  feedback message
- `class FooMultiturnAdapter(EnvAdapter)` — implement
  `build_followup_turn(history, last_prediction, instance) -> str`
  to produce the per-turn feedback message

The scaffold under `templates/inverse-problem/` does **not** generate
a multi-turn env by default; the brief is to start single-turn and add
the multi-turn variant only after the single-turn reward is solid.
