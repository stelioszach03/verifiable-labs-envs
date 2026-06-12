# verifiable-labs-tool-calling-multiturn

Multi-turn procedural tool-calling RL environment from the
Verifiable Labs catalogue. Same problem distribution as
`tool-calling-single`, with two changes:

1. **Verifier feedback between turns.** After each tool call the
   user message echoes the tool name + a short result preview +
   remaining budget. Hidden `gold_spec` is never serialised
   (R10 carry-over).
2. **Per-extra-turn penalty.** The first assistant turn is free;
   each additional one accrues 5% reward penalty, capped at 10%.

```
final_reward = base * (1 - min(0.05 · extra_turns, 0.10))
```

## Install

```bash
pip install verifiable-labs-tool-calling-multiturn
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
