# verifiable-labs-long-context-reasoning

Multi-hop long-context reasoning RL environment from the
Verifiable Labs catalogue. Each instance plants 2-3 chain facts +
1-2 distractors with similar surface form across distinct
documents in a procedurally generated multi-document corpus; the
model produces a structured answer (string or numeric), scored by
substring match for strings and by numeric tolerance (1 × 10⁻⁶)
for numbers.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing an `answer` field                    |
| `parse_valid`   | 0.20   | Extracted answer is non-empty                                            |
| `correctness`   | 0.70   | Exact / numeric match against the gold answer                            |

Three multi-hop templates ship in v0.0.1:

1. `chain_two_hop` — 2-hop fact composition (e.g. capital → population).
2. `chain_three_hop` — 3-hop transitive composition.
3. `arithmetic_over_facts` — 2-hop fact retrieval + simple arithmetic.

3 templates × 64-bit seed × 4 distractor-position modes × ~1e6
parameter combos — `EFFECTIVE_INSTANCES > 2e23`, well above the
contamination-resistance gate.

## Install

```bash
pip install verifiable-labs-long-context-reasoning
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
