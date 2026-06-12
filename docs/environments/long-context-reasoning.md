# `long-context-reasoning`

**Multi-hop chain QA with distractor needles.** Each instance
plants 2-3 chain facts + 1-2 distractor facts with similar surface
form (D4-C) across distinct documents in a procedurally generated
multi-document corpus. The model produces a structured answer
(string or numeric), scored by substring match for strings and
numeric tolerance ≤ 1 × 10⁻⁶ for floats (D3-A).

## Templates (D9 locked)

Three multi-hop templates ship in v0.0.1:

| Name                     | Hops | Gold-answer kind | Sample question shape                                                |
|--------------------------|------|------------------|----------------------------------------------------------------------|
| `chain_two_hop`          | 2    | numeric          | "What is the population of the capital of {Region}?"                 |
| `chain_three_hop`        | 3    | string           | "Who is the head of state of {Region}?"                              |
| `arithmetic_over_facts`  | 2    | numeric          | "What was the combined annual production of facilities A and B?"     |

Each instance carries:

- **2-3 chain facts** — the gold reasoning chain. Document IDs
  are recorded as `gold_chain_doc_ids` (oracle metadata; never
  shown to the model).
- **1-2 distractor facts** — same surface form (capital / mayor /
  production figure) but for a different entity. The model must
  filter on the entity name rather than pattern-matching the
  number.

## Schema

```json
{
  "answer": "12345",
  "confidence": 0.85
}
```

Numeric answers should be returned as plain numbers (no thousands
separators); string answers as the bare name.

## Reward decomposition

```
reward = 0.10 · format_valid    (output is parseable JSON
                                  with an `answer` field)
       + 0.20 · parse_valid     (extracted answer is non-empty)
       + 0.70 · correctness     (substring for string answers,
                                  numeric tolerance ≤ 1e-6 for floats)
```

The correctness branch dispatches on the instance's
`gold_answer_kind`:

- `gold_answer_kind == "numeric"` → first numeric token in the
  prediction is parsed and compared against the gold value with
  `abs(predicted - gold) <= 1e-6`.
- `gold_answer_kind == "string"` → substring + case-insensitive
  match (same comparator as `long-context-needle`).

## Procedural lattice

`EFFECTIVE_INSTANCES > 2 × 10²³` — 3 multi-hop templates × 64-bit
seed × 4 distractor-position modes × ~10⁶ parameter combos.

## Why the gold chain stays metadata

The `gold_chain_doc_ids` list (which documents carry the chain
facts) is **never** serialised into the prompt or the rewarded
output. v0.0.1 reward kernel uses `gold_answer` only; the chain
is reserved metadata for the v0.0.2 explainability layer (an aux
chain-reconstruction signal that doesn't bias the v0.0.1 reward
distribution).

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("long-context-reasoning")
inst = env.generate_instance(seed=42)
print(inst.template_name, inst.gold_answer_kind)
```

## See also

- [`long-context-needle`](long-context-needle.md) — single-turn
  retrieval baseline.
- [`long-context-synthesis`](long-context-synthesis.md) — 3-turn
  multi-needle synthesis with token-F1 feedback.
