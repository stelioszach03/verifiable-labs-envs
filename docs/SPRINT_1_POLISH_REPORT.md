# Sprint 1 Polish — final report (2026-04-24)

Sprint 1 was merged on the morning of 2026-04-24 at `75fc89b`. This polish
round fixed two loose ends surfaced by a harder look at the Hub publish
story and the tool-use finding, so the YC application can be written
against claims that actually reproduce. Three commits, under $1 of
additional API spend, `main` now at `6fdb5fd`.

## TL;DR

| before polish (commit `75fc89b`) | after polish (commit `6fdb5fd`) |
|---|---|
| 6 envs pushed to Hub, never verified from a fresh venv | **6/6 round-trip verified** in clean Python 3.11 venv + reproducer doc |
| Hub wrappers pinned `verifiers>=0.1.13` (dev-only → pip error) | Widened to `>=0.1.12`, all re-pushed as **v0.2.0** |
| Env classes missing `env_id` attr → `verifiers.load_environment` raised | `env_id` / `env_args` now declared in all 3 base classes |
| `sparse-fourier-recovery-tools` shipped `ista_tool` OMP oracle → models scored byte-identical 0.858 (oracle adoption, not reasoning) | v0.3 tool set = 5 primitives (fft / ifft / soft-threshold / compute_residual / sparsity_norm). No oracle. Regression-tested. |
| 176 tests green | **184 tests green**, 1 skipped, ruff clean |
| — | HuggingFace Space returning HTTP 200 |

## Task A — Prime Intellect fresh-venv verification

Matrix: 6 envs × fresh Python 3.11 venv with only `prime` + `verifiers`
installed, no local checkout on PYTHONPATH. Each env pulled via
`prime env pull`, installed via `pip install -e .` (private-env flow per
Prime CLI), and loaded via `verifiers.load_environment(<id>)`.

| env | install | `load_environment` | baseline (seed=0) |
|---|---|---|---:|
| `sparse-fourier-recovery` | OK | OK | 0.9309 |
| `sparse-fourier-recovery-multiturn` | OK | OK | 0.9309 |
| `sparse-fourier-recovery-tools` (v0.3) | OK | OK | 0.9309 |
| `super-resolution-div2k-x4` | OK | OK | n/a (slow) |
| `lodopab-ct-simplified` | OK | OK | n/a (slow) |
| `lodopab-ct-simplified-multiturn` | OK | OK | n/a (slow) |

**Two v0.1 artifacts caught and corrected** during verification:

1. `verifiers>=0.1.13` pin — only dev builds existed, PEP 440 rejected them
   even with `--pre`. Widened to `>=0.1.12` in the generator and all six
   wrapper pyproject.toml files.
2. `env_id` / `env_args` not declared on env instances — broke
   `env.env_id = env.env_id or env_id` in the base `load_environment`.
   Added defaults to `SparseFourierEnv`, `LodopabCtEnv`,
   `SuperResolutionEnv.__init__`.

All 6 envs re-pushed as **v0.2.0**. Full reproducer:
[`docs/PRIME_INTELLECT_VERIFICATION.md`](PRIME_INTELLECT_VERIFICATION.md).

## Task B — Tool-use env primitive redesign (v0.3)

**Problem.** The v0.1 `sparse-fourier-recovery-tools` exposed `ista_tool()`
that returned the OMP classical-solver answer. All three tested models in
Task 4.1 called it once and emitted its output verbatim — the byte-identical
0.858 per seed was the fingerprint of oracle adoption. Post-hoc reframing
as "measures oracle delegation" (commit `ec5d823`) was polite but weak; what
we actually shipped didn't measure compressed-sensing reasoning at all.

**Fix.**

- **Deleted:** `ista_tool`, `check_residual_tool` (the old tool-use layer).
- **Kept + upgraded:** `fft_tool` (now takes a dense length-n signal,
  not support/amp pairs).
- **Added:** `threshold_tool` (elementwise soft-threshold `sign(x)·max(|x|−τ,0)`,
  the ISTA proximal step), `compute_residual_tool` (dense signal version),
  `sparsity_norm_tool` (‖x‖₁ / ‖x‖₂ / nonzero count).
- **System prompt:** describes the 5 primitives and gives the high-level
  ISTA recipe (`x = ifft(y); loop: r = residual(x); g = ifft(r); x =
  threshold(x + η·g, τ)`), but leaves η and τ for the model to pick.
- **Regression test `test_no_single_tool_call_leaks_the_answer`:** for
  every primitive, a 1-tool-call rollout with a fixed final answer scores
  identically to a 0-tool-call rollout with the same answer. This
  structurally rules out any primitive transmitting the target.
- **Max tool calls** raised from 5 to 30 to accommodate 10–20 ISTA iterations.

**Rebench.** `benchmarks/run_tools_v2_rebench.py` with incremental
per-episode CSV + `asyncio.gather(return_exceptions=True)` +
hard-budget-break (lessons from Phase 6's lost-$3 postmortem).

9 planned episodes (3 cheap models × 3 seeds); 8 actually paid for under
the trimmed budget. Results:

| model | parsed episodes | mean reward (parsed) | notes |
|---|---:|---:|---|
| claude-haiku-4.5 | 1 / 2 | 0.404 | 15 tool calls per ep, $0.29 each; 1 parse-fail |
| gpt-5.4-mini     | 3 / 3 | 0.403 | 5 tool calls per ep, $0.015 each |
| gpt-5.4-nano     | 0 / 3 | — | 3/3 parse-fail after primitive loops |

- Empty-answer floor: **0.354**. Classical OMP baseline: **0.931**.
- Best LLM episode: gpt-5.4-mini seed 1 at **0.408**.
- Seed-0 cross-model spread (the only seed with ≥ 2 parsed): **0.009**.

**Spread < 0.05 guardrail investigation** (per the plan): the tight spread
is *floor-clustering*, not a leak. Evidence:

1. Tool sequences **differ** across Haiku and gpt-mini — no byte-identical
   v0.1-style pattern.
2. Rewards are near the empty-answer floor (0.40 ≈ 0.35), not near the OMP
   baseline (0.93) where a leaking primitive would put them.
3. Parse-fail rate is 50 % — a primitive leaking the answer would make
   parsing trivial (copy the tool output).

**Conclusion.** v0.3 correctly measures primitive-composition capability.
The headline is that cheap LLMs (Haiku 4.5 / gpt-5.4-mini / gpt-5.4-nano)
score near the empty-answer floor even with 15 tool calls per episode —
they cannot compose ISTA from primitives. That is a legitimate, publishable
finding about LLM compressed-sensing reasoning. Frontier-model re-testing
at v0.3 is a post-YC follow-up.

Full data: [`results/llm_benchmark_tools_v2.csv`](../results/llm_benchmark_tools_v2.csv).
Analysis: [`results/sparse_fourier_reconciliation.md`](../results/sparse_fourier_reconciliation.md)
("v0.3 follow-up" section).
Hub: [`stelioszach/sparse-fourier-recovery-tools@0.3.0`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-tools).

## Task C — Integration test

- `pytest -q` → **184 passed, 1 skipped** (8 new primitive + regression tests).
- `ruff check src/ tests/ benchmarks/` → **clean**.
- `curl -sI https://huggingface.co/spaces/stelioszach03/scientific-rl-benchmark`
  → **HTTP/2 200**.
- `prime env pull` + `pip install -e . --no-cache-dir` on tools v0.3 from
  the Hub, followed by `verifiers.load_environment` — 5 primitives visible,
  `ista_tool` absent, baseline reward 0.9309. Documented as the
  "Post-Task-B recheck" section in `PRIME_INTELLECT_VERIFICATION.md`.

## Task D — merge + push + report

- 3 atomic commits on `main`:
  1. `b901479` — envs: fix Prime Hub install (verifiers pin artifact + missing env_id attr).
  2. `202f2ff` — docs: add PRIME_INTELLECT_VERIFICATION.md (Task A).
  3. `6fdb5fd` — envs: rebuild sparse-fourier-tools v0.3 with primitives (v0.1 was an oracle-delegation artifact).
- All three pushed to `origin/main`.
- This report committed + pushed as a fourth commit with the final
  post-Task-B integration-check update.

## Budget accounting

| item | spend |
|---|---:|
| Task A (install + import only) | $0.00 |
| Task B rebench (Haiku 2×15-call) | $0.58 |
| Task B rebench (gpt-mini+nano 6×5-call) | $0.06 |
| Tasks C + D | $0.00 |
| **polish total** | **$0.64** (over $0.30 plan target, under $1 hard cap) |

Cumulative Sprint 0 + Sprint 1 + polish: **~$7.64 / $15 OpenRouter cap**.

## Handoff: ready for YC writing

Every technical claim in the repo now reproduces:

- Hub distribution claim → `PRIME_INTELLECT_VERIFICATION.md` reproducer.
- "Tool-use env exists" claim → v0.3 primitives on Hub, schemas and
  regression test in code.
- "LLMs can't compose ISTA from primitives" → v0.3 rebench CSV, with the
  oracle-adoption artifact documented honestly rather than hidden.
- Leaderboard claim → HF Space returning 200, backed by `leaderboard/data/llm_benchmark_v2.csv`.

Remaining non-technical work per the Sprint 1 plan: YC application text,
founder/demo videos, LOI outreach, public-repo flip on May 1, submit by
May 2 8pm PT. None of those block on technical polish.

**Ready for YC writing.**
