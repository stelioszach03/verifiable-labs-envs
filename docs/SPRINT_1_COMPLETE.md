# Sprint 1 Complete — 2026-04-24

Sprint 1 (YC-readiness depth pass) ran from 2026-04-23 to 2026-04-24 and
merged to `main` on completion. All 7 planned phases shipped.

## Final state

- **Branch**: `main` fast-forwarded through 12 Sprint 1 commits (`e1cd6f5` → `b0f8855`).
- **Tests**: **176 green**, 1 skipped (real-data gate on LoDoPaB slices).
- **Environments live**: 6 (3 single-turn + 2 multi-turn + 1 tool-use).
- **Sprint 1 incremental LLM spend**: $5.12 (within the $8 cap).
- **Cumulative OpenRouter spend**: ~$7.00 / $15 key-level cap.
- **Packages**: 7 verifiers-compatible subdirs under `packages/` (core + 6 envs).
- **Leaderboard**: Gradio app shipped under `leaderboard/`, HF Spaces deploy pending a `HF_TOKEN`.
- **Prime Intellect Hub**: package split shipped, `prime env push` pending interactive `prime login`.

## Phase-by-phase summary

### Phase 1 — Credibility polish
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 1.1 `docs/CONTAMINATION.md` + `scripts/memorization_probe.py` | `e1cd6f5` | +0 | $0.031 |
| 1.2 correlation analysis + failure taxonomy + `docs/METHODOLOGY.md` | `e6c131b` | +0 | $0.000 |

Memorization probe across Haiku 4.5, GPT-5.4-mini, and GPT-5.4-nano confirmed no constant-output patterns (cross-seed std 0.024–0.028 on sparse-Fourier). Cross-env Spearman correlation matrix (n=6 models) shows **SuperRes ↔ CT = +0.66** (same structural task), **SparseF ↔ image envs = −0.26 to −0.37** (different capabilities → the battery is non-redundant).

### Phase 2 — Real-data upgrade
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 2.1 `use_real_data=True` flag + LoDoPaB-CT chunked loader + download script | `d0bf160` | +7 | $0.013 |

Classical FBP on 10 real LoDoPaB-CT validation slices: mean reward **0.731** (predicted ~0.70 in the plan — landed dead-on). Claude Haiku 4.5 on 3 real-data seeds got 1/3 parse success with reward 0.694; 2/3 parse-failures ("expected 32 entries, got 31"), materially worse than its 0/5 phantom parse-fail rate. Real CT grids are genuinely harder for the model to transcribe than phantom ones — a legitimate discrimination signal for the v2 sweep.

### Phase 3 — Multi-turn environments
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 3.0 `LLMSolver.complete_turns` + `EnvAdapter.build_followup_turn` | `273c7ce` | +6 | $0.000 |
| 3.1 `sparse-fourier-recovery-multiturn` env + 13 tests | `0030321` | +14 | $0.000 |
| 3.1b async multi-turn sparse-F benchmark (3 models × 3 instances × 3 turns) | `82ed603` | +0 | $0.088 |
| 3.2 `lodopab-ct-simplified-multiturn` env + benchmark | `2b66e2e` | +13 | $0.562 |

Key finding from 3.1b: sparse-Fourier multi-turn **does not help** — models plateau or regress after turn 1 (residual feedback is informationally rich but models can't yet use it to revise sparse-support picks). This is itself a RLVR training signal. The CT multi-turn benchmark (3.2) showed the opposite: GPT-5.4 mean went from 0.601 single-turn to 0.654 multi-turn, Sonnet 0.580 → 0.640; Haiku and mini degraded, confirming "multi-turn helps frontier models, hurts small models" as a recurring pattern.

### Phase 4 — Tool-use environment
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 4.1 `sparse-fourier-recovery-tools` env + 4 tools (fft/ifft/ista/check_residual) + async benchmark | `eb55fc3` | +18 | $0.175 |

### Phase 5 — Prime Intellect Hub prep
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 5.1 + 5.2 7-package split under `packages/` + `docs/PRIME_INTELLECT.md` + auth-blocker log | `181ca57` | +0 | $0.000 |

Each package carries a `[project.entry-points."verifiers.environments"]` declaration so once `prime login` completes, `prime env push` per directory publishes each env to the Environments Hub. Full `prime` CLI attempt log at [`docs/prime_intellect_attempt.log`](prime_intellect_attempt.log).

### Phase 6 — Comprehensive v2 benchmark
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 6.1 `benchmarks/run_v2_benchmark.py` async sweep across 6 envs × 4 models × 2 instances | `aba364b` | +0 | $4.248 |

First attempt of this task lost $3.006 of API work to a bug in the `BudgetExceeded → asyncio.gather` control flow that discarded returned rows. The fix landed as incremental CSV writes after every episode; re-run with reduced scope (Opus dropped, 2 instances instead of 3) cost $1.242 and produced 78 rows. Full v2 table in the README and in [`results/benchmark_v2_summary.md`](../results/benchmark_v2_summary.md).

### Phase 7 — Static leaderboard
| Task | Commit | Tests Δ | Spend |
|---|---|---|---|
| 7.1 `leaderboard/` Gradio app + data bundle + `docs/LEADERBOARD.md` | `b0f8855` | +0 | $0.000 |

Deploys with `cd leaderboard && python app.py` locally (verified via import smoke). HF Spaces deploy pending `HF_TOKEN`.

## Key YC-narrative findings (v2 benchmark)

1. **Classical baselines beat every tested LLM** (~0.74 classical mean vs 0.49 best LLM mean). The environments are not saturated.
2. **Multi-turn helps frontier models on CT, hurts small models on CT** (Sonnet / GPT-5.4 +0.06, Haiku / GPT-5.4-mini −0.11 and −0.13 respectively).
3. **Sparse-Fourier stays flat across single / multi / tool-use rollout formats** (all ~0.33; v2 tools/single ratio = 1.03). Compressed sensing is not yet a text-completion task, full stop.

   > The earlier Task-4.1 number ("tool-use converges at 0.858, 2.4× single-turn") was a **scoring bug** — three different models produced byte-identical rewards because the benchmark was scoring the OMP baseline instead of the LLM's emitted answer. Full reconciliation: [`results/sparse_fourier_reconciliation.md`](../results/sparse_fourier_reconciliation.md). Any external material (YC app, README) must use the v2 numbers.
4. **SuperRes saturates for the frontier cluster** at ~0.72–0.73; GPT-5.4-mini trails at 0.53 (budget models can't match the big cluster on image denoising).
5. **Parse-failure rate scales inversely with model size** on long-JSON grid outputs — holds from Sprint 0 through v2.
6. **Real CT is genuinely harder than phantom CT** for LLM transcription — the 2/3 parse-fail rate spike on Haiku is the evidence.
7. **Residual feedback in the multi-turn format is not a uniform win** — it helps only models that can maintain coherence across the 3-turn conversation, which is itself a capability axis.
8. **Procedural regeneration defeats fixed-string memorization** structurally, not empirically — the memorization probe merely confirms that the pipeline actually responds to its input (cross-seed std ≥ 0.02 on every tested model).

## Remaining non-technical work

- Finalize the YC application writing (text lives in `~/Documents/yc-s26/application.md`) — update with the v2 benchmark numbers + new finding bullets.
- Record founder + demo videos per the scripts in `~/Documents/yc-s26/`.
- Cofounder search — still deferred per Sprint 0's "solo is fine for now" stance.
- LOI outreach to 1–2 candidate design partners (radiology AI or national-lab AI groups).
- Flip the GitHub repo to public on **May 1**.
- Submit the YC application on or before **May 2**, 8pm PT.
- One-time ops: `prime login` + `huggingface-cli login` to unblock the two pending publish paths.

## Budget accounting

- Sprint 0 (shipped 2026-04-23): $1.890 (v1 benchmark + Day-2/3/4/5 polish).
- Sprint 1 today: **$5.112** incremental.
    - Phase 1: $0.031
    - Phase 2: $0.013
    - Phase 3: $0.650 (3.1b $0.088, 3.2 $0.562)
    - Phase 4: $0.175
    - Phase 5: $0
    - Phase 6: $4.248 (first attempt lost $3.006, re-run $1.242)
    - Phase 7: $0
- **Cumulative OpenRouter spend: ~$7.00 / $15 key-level cap**.
