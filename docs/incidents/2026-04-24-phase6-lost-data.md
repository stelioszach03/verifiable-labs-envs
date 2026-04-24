# Incident: Phase 6 v2 benchmark lost $3.01 of completed API calls

## Timeline

- **2026-04-24, Phase 6 task 6.1 launch**: comprehensive v2 benchmark fired with `--max-cost 3.0` across 6 models × 6 envs × 3 instances.
- **~230 s in**: the `Budget` guard reported cumulative spend $3.0061, crossing the cap.
- `BudgetExceeded` raised inside the task that tripped the guard; exception propagated through the wrapping `await asyncio.gather(*tasks, return_exceptions=False)`.
- `gather` cancelled pending tasks AND discarded the return values of already-completed tasks.
- `nested = await asyncio.gather(...)` was never assigned, so the subsequent `for rows in nested: all_rows.extend(rows)` loop never ran.
- Final state: **0 rows persisted** to CSV, **$3.0061 spent**, no data recovered.

## Root cause

Three compounding design choices:

1. **`return_exceptions=False`** — the default for `asyncio.gather` re-raises the first exception and discards all other results, including completed successes.
2. **End-of-run CSV write** — rows were collected in a shared `all_rows` list only AFTER `gather` returned; the incremental `append` was not happening per-task.
3. **Budget check inside task body** — `budget.add(cost)` ran after each API call *inside* the task. By the time it tripped, the API money was already spent and the completed row was only in that task's local `rows`.

Any two of these could coexist safely. All three together meant "when the cap fires, you lose everything already paid for."

## Fix

The v2 rerun and every future benchmark in this repo now uses:

- `return_exceptions=True` on `asyncio.gather` so completed tasks' data is preserved even if siblings raise.
- **Incremental CSV append** inside a per-task wrapper (`_episode_and_persist` in `benchmarks/run_v2_benchmark.py`) so every finished episode writes to disk before the next API call resolves.
- Budget still checked inside the task (we want the cap to fire ASAP), but the wrapper catches `BudgetExceeded` locally, marks `aborted=True` on the shared closure, and lets already-persisted rows survive.

Rerun on a reduced scope (4 models instead of 6, 2 instances instead of 3) recovered 78 rows at a further $1.24 cost, producing the v2 table referenced in the README.

## Lessons for future benchmarks in this repo

1. **Default `asyncio.gather(..., return_exceptions=True)`** for any benchmark harness.
2. **Write-on-completion**, never write-at-end, for data you're paying for.
3. **Budget enforcement happens at task dispatch**, not only inside the task. Check `budget.remaining > estimated_call_cost` BEFORE launching a new coroutine.
4. **Pre-flight cost estimate** printed before launch (existing `--dry-run` does this — enforce running it in a CI smoke test).
5. **Post-run invariant**: `rows_persisted_to_csv == api_calls_that_returned_successfully`. If these disagree, the harness is buggy.

## Spend accounting

- Phase 6 first attempt: **$3.0061** real API spend, **0** rows saved.
- Phase 6 rerun (fixed harness, reduced scope): **$1.2416** real API spend, **78** rows saved.
- Phase 6 total: **$4.2477** spent, 78 rows in `results/llm_benchmark_v2.csv`.
- No impact on end-of-sprint deliverables: the v2 heatmap, summary, and README table all come from the 78-row rerun.

## Non-impact

- No tests ship with a dependency on the lost first-run data.
- No document external to Sprint 1 references the lost data.
- OpenRouter key still well under cap ($7 / $15 total after both runs).
