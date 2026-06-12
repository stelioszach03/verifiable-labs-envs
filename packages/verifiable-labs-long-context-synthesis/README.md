# verifiable-labs-long-context-synthesis

Multi-needle 3-turn long-context synthesis RL environment from
the Verifiable Labs catalogue. Each instance carries 3-5 needles
spread across distinct documents in a procedurally generated
multi-document corpus; the model produces a free-text answer
combining all needles, and is scored by SQuAD-style token-F1
against the gold concatenation.

The rollout is **3-turn** with feedback between turns:

```
Turn 1  →  context blob + question                  →  answer_v1
Turn 2  →  feedback (F1 score + needle doc indices) →  answer_v2
              (NO gold answer text)
Turn 3  →  same                                     →  answer_final
```

Final reward applies the standard turn-count penalty (5%/turn,
cap 10%) — same constants as the math/code multi-turn families.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing an `answer` field                    |
| `parse_valid`   | 0.20   | Extracted answer is non-empty                                            |
| `correctness`   | 0.70   | Token-F1 against the concatenated gold facts                             |

10 procedural topic templates × 64-bit seed × 3 count modes ×
~1e6 parameter combos — `EFFECTIVE_INSTANCES > 5e23`, well above
the contamination-resistance gate.

## Install

```bash
pip install verifiable-labs-long-context-synthesis
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
