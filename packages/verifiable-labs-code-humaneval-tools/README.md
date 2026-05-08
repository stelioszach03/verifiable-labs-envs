# verifiable-labs-code-humaneval-tools

Tool-use procedural code-execution RL environment from the
Verifiable Labs catalogue. Same problem distribution as
`code-humaneval`, but the model converges via **three primitives**
instead of returning a single completion:

| Tool         | Purpose                                                                  |
|--------------|--------------------------------------------------------------------------|
| `read_file`  | Read a file in the per-call workspace (`solution.py`, `test_solution.py`) |
| `write_file` | Overwrite a workspace file (typically `solution.py`)                      |
| `run_test`   | Run a single visible pytest case in the D5 sandbox                        |

The workspace is seeded with an empty `solution.py` and a
`test_solution.py` containing the **visible tests only** — the
hidden test suite is the held-out grading signal (R10).

`DEFAULT_MAX_TOOL_CALLS = 30`, `DEFAULT_TOOL_TIMEOUT_S = 5.0` per
tool invocation.

## Install

```bash
pip install verifiable-labs-code-humaneval-tools
```

Source of truth + full docs:
https://github.com/stelioszach03/verifiable-labs-envs
