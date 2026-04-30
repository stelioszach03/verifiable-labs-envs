# vlabs-calibrate examples

Three self-contained demos showing how to wrap external reward functions
with conformal coverage guarantees. None of these examples depend on the
`verifiable-labs-envs` envs — every demo synthesises its calibration data
inline so you can run each script with zero configuration.

| file | reward shape | non-conformity | what it proves |
|---|---|---|---|
| `01_humaneval_passfail.py` | binary 0/1 (test-case pass) | `binary` | API works for pass/fail rewards; surfaces the binary-degeneracy caveat |
| `02_math_exact_match.py` | binary 0/1 with judge confidence as σ | `scaled_residual` | exact-match style task with smooth uncertainty estimate |
| `03_gsm8k_step_validity.py` | continuous in `[0, 1]` | `scaled_residual` | continuous reward + ensemble-style σ |

Run any one of them:

```bash
python examples/calibrate/01_humaneval_passfail.py
```

Each script prints a coverage table and exits 0 when it hits its
`(1 − α) ± 5pp` target coverage. Scripts are intentionally short
(~80–120 LOC each) so you can copy a snippet into your own pipeline.
