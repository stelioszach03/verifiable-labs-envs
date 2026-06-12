# `long-context-synthesis`

**Multi-needle 3-turn synthesis with token-F1 scoring.** Each
instance plants 3-5 needles (D4-B) across distinct documents in
a procedurally generated multi-document corpus; the model produces
a free-text answer combining all needles, scored by SQuAD-style
token-F1 against the concatenation of the gold facts.

## Rollout

| Turn | What the model sees                                      | What it returns |
|------|----------------------------------------------------------|-----------------|
| 1    | Context blob + question                                  | `answer_v1`     |
| 2    | Feedback (F1 score + needle doc indices, NO gold answer) | `answer_v2`     |
| 3    | Same                                                     | `answer_final`  |

The gold answer string is **never** shown to the model (R10
carry-over). The inter-turn feedback is bucketed:

```
F1 < 0.50  →  "covers ~X%; review docs at indices [...] for missing facts"
F1 < 0.90  →  "covers ~X%; refine wording or add missing facts"
F1 ≥ 0.90  →  "previous answer is largely correct; you may keep it"
```

## Turn-count penalty

```
final_reward = base × (1 − min(0.05 · (n_turns − 1), 0.10))
```

Three turns scores 0.9× the equivalent single-turn reward — same
constants as `code-humaneval-multiturn` / `math-algebra-multiturn`
(D6-A locked across multi-turn families).

## Schema

```json
{
  "answer": "production figure ABCD-1234; verified balance WXYZ-5678; identifier QRST-9999",
  "confidence": 0.85
}
```

## Reward decomposition

```
reward = 0.10 · format_valid    (output is parseable JSON
                                  with an `answer` field)
       + 0.20 · parse_valid     (extracted answer is non-empty)
       + 0.70 · correctness     (D3-C: SQuAD-style token-F1 against
                                  the concatenated gold facts)
```

Token-F1 is whitespace-tokenised, lowercased, punctuation-stripped.
The correctness term is graded — it returns a float in `[0, 1]` rather
than a 0/1 indicator. This is the only env in the catalogue with
graded correctness; needle / reasoning envs use exact match.

## Procedural lattice

`EFFECTIVE_INSTANCES > 5 × 10²³` — 10 topic templates × 64-bit
seed × 3 needle-count modes (3 / 4 / 5 needles) × ~10⁶ parameter
combos.

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("long-context-synthesis", max_turns=3)
inst = env.generate_instance(seed=42)
out = env.run_rollout(solver, inst)  # solver is your LLMSolver
print(out["meta"]["turn_rewards"], out["meta"]["turn_penalty"])
```

## See also

- [`long-context-needle`](long-context-needle.md) — single-turn
  retrieval baseline.
- [`long-context-reasoning`](long-context-reasoning.md) — multi-hop
  chain QA with distractor needles.
