# verifiable-labs-long-context-needle

Single-turn long-context needle-in-haystack RL environment from
the Verifiable Labs catalogue. Each instance hands the model a
procedurally generated multi-document corpus + a question, and
expects a JSON envelope with an `answer` field. The needle (the
answer-bearing sentence) is injected at one of four positions
(`start | middle | end | random`); the verifier scores by
substring match against the needle's distinctive token.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing an `answer` field                    |
| `parse_valid`   | 0.20   | Extracted answer is non-empty                                            |
| `correctness`   | 0.70   | Substring match against the gold needle (case-insensitive — D3-A)        |

10 procedural topic templates × 64-bit seed × 4 position modes ×
~1e6 parameter combos — `EFFECTIVE_INSTANCES > 7.4e23`, well above
the contamination-resistance gate.

## Install

```bash
pip install verifiable-labs-long-context-needle
```

Source of truth + full docs:
https://github.com/stelioszach03/verifiable-labs-envs
