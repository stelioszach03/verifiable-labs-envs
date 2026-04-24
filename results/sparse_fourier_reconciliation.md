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
