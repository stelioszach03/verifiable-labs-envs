# `code-humaneval`

**Single-turn procedural code-execution with sandboxed pytest scoring.**
Given a function signature + docstring + visible test block, the
solver returns a Python implementation. The env runs the
implementation against a hidden test suite inside a D5-bounded
subprocess sandbox and scores the pass-rate.

This is the first member of the **code-execution** template family
introduced in Phase 24.

## Problem

| | |
|---|---|
| input | HumanEval-shaped prompt (signature + docstring + 2–3 visible asserts) |
| output | JSON `{"code": "<Python source>", "confidence": <float in [0, 1]>}` |
| gold | hidden test battery (5–10 asserts) — pass-rate becomes the reward |
| sandbox | 512 MB / 30 s wall / 20 s CPU / `unshare -rn` net / 16-proc cap |

Twelve procedural templates spanning lists, strings, dicts, ints,
trees, and graphs:

| # | Template | Sample task |
|---|---|---|
| 1 | `list_sum_filter` | sum elements above a threshold |
| 2 | `list_two_sum` | return indices summing to target |
| 3 | `list_running_max` | running maximum |
| 4 | `string_reverse_words` | reverse word order, collapse whitespace |
| 5 | `string_count_substring` | overlapping substring count |
| 6 | `string_palindrome_check` | palindrome boolean (alnum, case-insensitive) |
| 7 | `dict_invert` | invert keys ↔ values |
| 8 | `dict_merge_with_resolver` | merge with strategy sampled per seed |
| 9 | `int_digit_root` | repeated decimal-digit-sum |
| 10 | `int_factor_count` | count positive divisors |
| 11 | `tree_node_count_leaves` | leaf count on a procedurally generated tree |
| 12 | `graph_shortest_path` | BFS distance on adjacency dict |

Procedural-regeneration certification: 12 templates × 64-bit seed ×
~10⁶ parameter combinations ≈ 7.4 × 10²³ effective instances, well
above the platform's 1 × 10¹⁵ contamination-resistance gate.

## Variants

- [`code-humaneval`](#) — single-turn, this page.
- [`code-humaneval-multiturn`](code-humaneval-multiturn.md) — 3-turn
  rollout with visible-test feedback between turns.
- [`code-humaneval-tools`](code-humaneval-tools.md) — tool-use loop
  with `read_file` / `write_file` / `run_test` primitives.
- [`code-mini-repo`](code-mini-repo.md) — repo-scale variant
  (3-file synthetic mini-repo, multi-file edits).

## Schema

```json
{
  "code": "def solve_list_sum_filter(nums, threshold):\n    return sum(n for n in nums if n > threshold)",
  "confidence": 0.85
}
```

## Reward decomposition

```
reward = 0.10 * format_valid    (output is parseable JSON
                                  containing a `code` field)
       + 0.20 * parse_valid     (extracted code compiles via
                                  compile(..., "exec"))
       + 0.70 * pass_rate       (passes / total over visible ∪ hidden)
```

Pass rate is the continuous signal that conformal calibration needs
(R7 in PHASE_24_PLAN.md). Identical weight structure to the
`math-algebra` family for cross-env comparability.

## Sandbox guarantees

The pytest invocation runs through
`verifiable_labs_envs.sandbox.execute_in_sandbox_sync`. PHASE_24_PLAN.md
§6 locks the D2-A "subprocess + rlimit" mechanism under a
**trusted-input scope** — the locked guarantee is *isolation between
concurrent customer calls on the same machine*, not *defence against a
determined attacker*. A future public anonymous-submit endpoint would
require flipping the sandbox to D2-B (Docker) or D2-C (Firecracker);
the upgrade-gate sentinel test in `tests/test_sandbox.py` keeps that
gate visible.

| Surface | Guarantee | Sentinel test |
|---|---|---|
| Filesystem | No write outside per-call tmpdir | `test_sandbox_cannot_write_outside_tmp` |
| Network | Outbound socket connect fails (`unshare -rn`) | `test_sandbox_no_network_blocks_socket` |
| CPU | `while True` killed within `RLIMIT_CPU + 2 s` | `test_sandbox_cpu_timeout_kills_busy_loop` |
| Memory | `bytearray(900 << 20)` over cap killed | `test_sandbox_memory_cap_kills_oom` |
| Wall-clock | `time.sleep(60)` killed at ≤ 32 s | `test_sandbox_wall_timeout_kills_long_running` |
| Process fanout | Fork bomb hits `RLIMIT_NPROC` | `test_sandbox_proc_cap_blocks_fork_bomb` |
| Cleanup | tmpdir wiped on every exit path | `test_sandbox_tmpfs_cleanup_after_call` |

## Loading

```python
from verifiable_labs_envs import load_environment

env = load_environment("code-humaneval")
inst = env.generate_instance(seed=42)
print(inst.prompt)
```

Or pin a calibration quantile to skip the auto-calibration sweep:

```python
env = load_environment("code-humaneval", calibration_quantile=0.5)
```

## Tests

The repo's `tests/test_code_humaneval.py` runs unit-level reward
checks; `tests/test_sandbox.py` runs the D5-limit sentinels. Both
require Linux (the rlimit + `unshare -rn` primitives are POSIX-only).
