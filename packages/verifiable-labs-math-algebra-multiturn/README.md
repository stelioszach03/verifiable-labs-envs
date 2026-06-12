# verifiable-labs-math-algebra-multiturn

Multi-turn algebraic-simplification RL environment with SymPy-verified rewards, conformal coverage, and verifier feedback.

Three-turn variant of [`verifiable-labs-math-algebra`](../verifiable-labs-math-algebra/):

1. **Turn 1**: the LLM sees the problem and proposes an answer + confidence.
2. **Turn 2**: the env returns verifier feedback — whether the previous answer
   was correct, and if not, at which validation step it failed
   (format → parse → equivalence). The gold expression itself is never revealed.
3. **Turn 3**: same — final answer.

Reward is computed on the **final** turn's prediction with a turn-count
penalty `(1 − 0.05 · (n_turns − 1))` capped at `0.10`, so 3 turns scores
0.9× the equivalent single-turn reward. Per-turn trajectory exposed in
`meta.turn_rewards` / `meta.turn_components` for the v2 benchmark.

## Install

```
pip install "git+https://github.com/verifiablelabs/verifiable-labs-envs.git@main#subdirectory=packages/verifiable-labs-math-algebra-multiturn"
```

## Use

```python
from verifiable_labs_math_algebra_multiturn import load_environment

env = load_environment(calibration_quantile=0.5)
inst = env.generate_instance(seed=0)
print(inst.prompt)
# At inference time, the env's run_rollout() drives the 3-turn dialogue.
```

## Why the multi-turn variant

A single-turn variant scores the model's ability to one-shot the
answer; the multi-turn variant additionally scores the model's
ability to USE feedback. Many real-world math agents (proof search,
constraint solvers) iterate against a verifier — this env captures
that loop while keeping the gold expression hidden so the model
cannot trivially mimic it.
