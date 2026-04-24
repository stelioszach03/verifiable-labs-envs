# Sparse-Fourier rollout-format reconciliation

Task 4.1 (dedicated tool-use benchmark, phase 4) reported a tool-use mean reward of ≈0.858 on sparse Fourier, about **2.4×** the single-turn Sprint-0 baseline. Phase 6 v2 (comprehensive sweep) reported that sparse-Fourier stays **flat** across single-turn, multi-turn, and tool-use rollout formats. Both claims can't be right. This note picks a verdict from the raw numbers.

## Per-model table

| model | single (v2) | multi (v2) | tools (v2) | tools (v1 = Task 4.1) | Δ v2 tools − single |
|---|---:|---:|---:|---:|---:|
| claude-haiku-4.5 | 0.364 (n=2) | 0.351 (n=2) | 0.334 (n=2) | 0.858 (n=3) | -0.030 |
| claude-sonnet-4.6 | 0.305 (n=2) | 0.328 (n=2) | 0.337 (n=2) | 0.858 (n=3) | +0.032 |
| gpt-5.4 | 0.293 (n=2) | 0.365 (n=2) | 0.306 (n=1) | — | +0.013 |
| gpt-5.4-mini | 0.338 (n=2) | 0.363 (n=2) | 0.354 (n=1) | 0.858 (n=3) | +0.016 |

## Overall means

- v2 single-turn: **0.325** (n=8)
- v2 tool-use:   **0.334** (n=6)
- v1 tool-use (Task 4.1): **0.858** (n=9)

- **v2 tools ÷ v2 single:** 1.03
- **v1 vs v2 tools (abs diff):** +0.525

## Verdict: **B**

TOOL-USE CONVERGENCE DID NOT REPLICATE — v2 tools mean is within 0.009 of single-turn mean. Task 4.1's '2.4× improvement' was likely a small-sample artifact (N≈3 instances per model).

## Statistical caveat

The Phase 6 v2 sweep used N=2 instances per (model, env) pair; Task 4.1 used N=3 instances per model. Neither sample is large enough to support a tight confidence interval — these numbers are *descriptive*, not significance-tested. The verdict is chosen on the size of the observed mean gap vs the per-cell standard deviation.

## What to do in external documents

- **Remove any '2.4× improvement' claim from the YC application and README.**
- Keep the tool-use env in the product (it still provides a richer action surface for RLVR training), but don't headline it as a *reward* boost.
- Headline findings that survived v2:
  * Classical baselines beat every tested LLM (~0.74 vs 0.49 mean).
  * Multi-turn helps frontier models on CT, hurts small models on CT.
  * Parse-failure rate scales inversely with model size on long-JSON grid outputs.
  * Real LoDoPaB-CT is harder to transcribe than phantom CT for the same model.

## v0.3 follow-up — primitive-composition rebench (2026-04-24, Sprint 1 Polish)

The v0.1/v0.2 tool set contained an ``ista_tool`` that returned the classical
OMP reconstruction directly. On inspection, all three tested models in
Task 4.1 simply called it once and copied its output verbatim — the
byte-identical scores in the v1 table are the fingerprint of that shortcut.
That made Task 4.1 a measurement of *oracle adoption*, not reasoning.

v0.3 ships a primitive tool set (``fft_tool``, ``ifft_tool``,
``threshold_tool`` soft-threshold, ``compute_residual_tool``,
``sparsity_norm_tool``) with the ``ista_tool`` oracle removed. A regression
test (`test_no_single_tool_call_leaks_the_answer`) verifies that any single
primitive call followed by a fixed final answer scores identically to that
final answer with 0 tool calls — i.e. no primitive transmits the target to
the model.

Rebench on seeds 0–2 under a $0.30–1.00 budget corridor (tightened from the
original $0.30 target after Haiku's 15-tool-call sweep spent $0.29 per
episode):

| model | seed | tools | reward | parse_ok |
|---|---:|---:|---:|:---|
| claude-haiku-4.5 | 0 | 15 | 0.404 | OK |
| claude-haiku-4.5 | 1 | 15 | — | FAIL |
| gpt-5.4-mini     | 0 | 5  | 0.395 | OK |
| gpt-5.4-mini     | 1 | 5  | 0.408 | OK |
| gpt-5.4-mini     | 2 | 5  | 0.407 | OK |
| gpt-5.4-nano     | 0 | 5  | — | FAIL |
| gpt-5.4-nano     | 1 | 5  | — | FAIL |
| gpt-5.4-nano     | 2 | 3  | — | FAIL |

- Empty-answer floor (no tools, zeros everywhere): reward ≈ **0.354**.
- Classical OMP baseline: reward ≈ **0.931**.
- Best-case v0.3 LLM rollout: reward **0.408** (gpt-5.4-mini, seed 1).

**Cross-model spread on seed 0** (only seed with ≥ 2 parsed rewards):
|0.404 − 0.395| = **0.009**. This is below the ≥ 0.08 investigation target,
so per the Sprint 1 Polish plan we investigated:

1. **Tool sequences differ** across Haiku / gpt-mini
   (`ifft, compute_residual, ifft, threshold, …` vs.
   `ifft, compute_residual, sparsity_norm, …`).
2. **Rewards cluster near the empty-answer floor** (0.40 vs floor 0.35),
   not near the OMP baseline (0.93). If a primitive were leaking the
   answer, rewards would be near 0.93 as in v0.1.
3. **Parse-fail rate is 50%** across the 8-episode matrix — half the models
   can't emit a valid final JSON after the tool loop. An oracle-leaking
   primitive would make parsing trivial (copy the tool output).

Conclusion: the tight spread is **floor-clustering due to genuine LLM
incapacity**, not a hidden oracle. v0.3 correctly measures
primitive-composition capability; the headline is that current cheap LLMs
cannot yet compose ISTA from five primitive operators even with 15 tool
calls per episode. That is itself a legitimate publishable finding about
compressed-sensing reasoning.

Full data: [`results/llm_benchmark_tools_v2.csv`](llm_benchmark_tools_v2.csv).
v0.3 push to the Hub at
[`stelioszach/sparse-fourier-recovery-tools@0.3.0`](https://app.primeintellect.ai/dashboard/environments/stelioszach/sparse-fourier-recovery-tools).
Rebench script: [`benchmarks/run_tools_v2_rebench.py`](../benchmarks/run_tools_v2_rebench.py).

## Updated headline findings (after v0.3 rebench)

1. Classical baselines beat every tested LLM on the image-and-CT envs
   (~0.74 vs 0.49 mean).
2. Multi-turn helps frontier models on CT, hurts small models on CT.
3. Parse-failure rate scales inversely with model size on long-JSON grid outputs.
4. Real LoDoPaB-CT is harder to transcribe than phantom CT for the same model.
5. **New:** On the primitive-composition tool-use env, cheap LLMs
   (Haiku-4.5 / gpt-5.4-mini / gpt-5.4-nano) score near the empty-answer
   floor — they cannot compose ISTA from primitives. Frontier-model
   re-testing at v0.3 is a post-YC follow-up.
