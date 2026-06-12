# verifiable-labs-code-humaneval-multiturn

Multi-turn procedural code-execution RL environment from the
Verifiable Labs catalogue. Same problem distribution as
`code-humaneval`, but the model gets up to **3 turns** with
visible-test feedback between them.

| Turn | What the model sees                                                         | What it returns          |
|------|------------------------------------------------------------------------------|--------------------------|
| 1    | Function signature + docstring + visible test block                          | First implementation     |
| 2    | Visible test pass/fail counts (no test source, no oracle)                    | Revised implementation   |
| 3    | Same — final attempt scored against visible ∪ hidden tests                   | Final implementation     |

A **turn-count penalty** of 5% per extra turn (capped at 10%) keeps
multi-turn from being a free win — three turns scores 0.9× the
equivalent single-turn reward. Hidden tests are **never** shown to
the model (R10 — visible test pass count is the only feedback signal).

## Install

```bash
pip install verifiable-labs-code-humaneval-multiturn
```

Source of truth + full docs:
https://github.com/verifiablelabs/verifiable-labs-envs
