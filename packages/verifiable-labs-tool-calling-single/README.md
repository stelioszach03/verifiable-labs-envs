# verifiable-labs-tool-calling-single

Single-pass procedural tool-calling RL environment from the
Verifiable Labs catalogue. Each instance hands the model a
natural-language goal plus a subset of five mock primitives
(`calculator`, `web_search`, `read_file`, `write_file`,
`send_message`), and scores the resulting trajectory.

| Component       | Weight | What it rewards                                                          |
|-----------------|--------|---------------------------------------------------------------------------|
| `format_valid`  | 0.10   | Final non-tool message is parseable JSON                                  |
| `parse_valid`   | 0.20   | Every tool-call carried valid args AND the final submission parses        |
| `correctness`   | 0.70   | D2-C blend: 0.30 · action_validity + 0.70 · final_state_match             |

10 procedural templates spanning arithmetic, search-and-email,
file-concat, multi-search, outbox-audit and more —
`EFFECTIVE_INSTANCES > 6e23`, well above the contamination-resistance
gate.

## Install

```bash
pip install verifiable-labs-tool-calling-single
```

## Use

```python
from verifiable_labs_tool_calling_single import load_environment

env = load_environment(calibration_quantile=0.5)
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

Source of truth + full docs:
https://github.com/stelioszach03/verifiable-labs-envs
