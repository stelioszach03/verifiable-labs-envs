# verifiable-labs-tool-calling-debug

Trace-debug procedural tool-calling RL environment from the
Verifiable Labs catalogue. PHASE_25_PLAN.md D8-C — given a partial
trajectory + the workspace state it produced, the model continues
the rollout and submits a final answer.

Three templates wrap base single-turn templates:

| Debug template       | Base                  | Prefix supplied                                     |
|----------------------|-----------------------|------------------------------------------------------|
| `partial_compute`    | `arithmetic_compute`  | First `(a+b)` step pre-computed; model finishes.    |
| `partial_search`     | `search_and_email`    | `web_search` call done; model sends the email.      |
| `partial_workspace`  | `file_concat`         | Both files read into state; model writes the merge. |

Reward kernel reuses the single-turn shape verbatim — same
`format_valid` + `parse_valid` + `correctness` weights, same D2-C
composite. The conformal layer is independent (separate calibration
quantile to reflect the debug-shape residual distribution).

## Install

```bash
pip install verifiable-labs-tool-calling-debug
```

Source of truth + full docs:
https://github.com/stelioszach03/verifiable-labs-envs
