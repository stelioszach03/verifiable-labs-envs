# v2 benchmark summary

Source: [`results/llm_benchmark_v2.csv`](llm_benchmark_v2.csv). Each cell is the mean final-turn reward across seeds for one (model, env) pair. Dashes mean no successful row was recorded (either not attempted in this sweep or every seed parse-failed).

## Table

| model | SparseF | SparseF-MT | SparseF-Tools | SuperRes | CT | CT-MT | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-4.5 | 0.364 | 0.351 | 0.334 | 0.726 | 0.640 | 0.527 | **0.490** |
| claude-sonnet-4.6 | 0.305 | 0.328 | 0.337 | 0.726 | 0.580 | 0.640 | **0.486** |
| gpt-5.4 | 0.293 | 0.365 | 0.306 | 0.721 | 0.601 | 0.654 | **0.490** |
| gpt-5.4-mini | 0.338 | 0.363 | 0.354 | 0.534 | 0.505 | 0.371 | **0.411** |
| **env mean** | 0.325 | 0.352 | 0.333 | 0.677 | 0.581 | 0.548 | |

## Notes

- Multi-turn and tool-use envs report the LAST successful turn's reward. If every turn parse-failed, no row lands here. The full per-turn trajectory is in the CSV.
- Multi-turn envs do not always help — see the per-row `turn_rewards` in `meta` for plateau / regression patterns.
- The tool-use env relies on the adapter's `execute_tool_call` dispatch; a turn whose text is neither a valid final-answer JSON nor a valid tool call is recorded as a parse failure and ends the episode.
- Opus 4.7 was dropped from this v2 sweep because Sprint-0 + Sprint-1 partial runs showed Sonnet ≈ Opus within noise on these envs; keeping it in would have blown the $3 cap per the plan's mitigation ladder.