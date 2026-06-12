# verifiable-labs-code-humaneval

Single-turn procedural code-execution RL environment from the
Verifiable Labs catalogue. Each instance carries a function signature
+ docstring + a small visible test set; the model returns Python
source which the env runs against a hidden test battery inside a
sandboxed subprocess (D5 limits: 512 MB / 30 s wall / 20 s CPU /
``unshare -rn`` network isolation / 16-process fanout cap).

| Component       | Weight | What it rewards                                          |
|-----------------|--------|----------------------------------------------------------|
| `format_valid`  | 0.10   | Output is parseable JSON containing a `code` field       |
| `parse_valid`   | 0.20   | Extracted code compiles via `compile(..., "exec")`       |
| `pass_rate`     | 0.70   | Fraction of (visible ∪ hidden) pytest cases that passed  |

12 procedural templates across lists, strings, dicts, ints, trees,
and graphs — `EFFECTIVE_INSTANCES > 7e23`, well above the
contamination-resistance gate.

## Install

```bash
pip install verifiable-labs-code-humaneval
```

## Use

```python
from verifiable_labs_code_humaneval import load_environment

env = load_environment(calibration_quantile=0.5)
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
