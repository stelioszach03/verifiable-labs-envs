# Verifiable Labs — one-pager

**Verifiable Labs is the API, SDK, and CLI layer for evaluating and
training scientific AI agents on verifiable RL environments.**

---

## Problem

Frontier models are increasingly trained on **verifiable rewards**
(RLVR). Today's RL environments are mostly text-only, saturate
quickly under memorisation, and miss the **continuous,
uncertainty-sensitive reasoning** real science requires —
imaging, signal recovery, physics inversion. Standard LLM eval tools
(LM Eval Harness, Ragas, OpenAI Evals, …) are good for chatbot apps,
but cannot drive scientific RL training because they:

- score against static answer keys (memorisable),
- do not measure calibrated uncertainty,
- offer no procedural problem regeneration,
- and have no classical-baseline reference to ground the difficulty.

The result: scientific AI teams either run paper-style benchmarks
once a quarter, or build their own ad-hoc evaluation harness.

## Solution

A platform that *generates* verifiable RL environments for scientific
domains, and exposes them through three coherent surfaces:

| surface | use case |
|---|---|
| **REST API** (`api.verifiable-labs.com`) | language-agnostic eval; CI gates |
| **Python SDK** (`pip install verifiable-labs`) | scripted access; sync + async |
| **`verifiable` CLI** (`pip install verifiable-labs-envs`) | local eval, JSONL traces, Markdown reports |

A single `verifiable run` writes a JSONL trace conforming to a stable
schema. `verifiable report` turns it into a 12-section Markdown
evaluation. `verifiable compare` does side-by-side comparisons.
Custom envs are scaffolded with `verifiable init-env` and validated
with `verifiable validate-env`.

## Why now

1. **RLVR is going mainstream.** The recipe (rollout → verifiable
   reward → policy update) needs verifiable environments, and the
   industry has converged on text + code so far. Scientific RL is
   the unmet wedge.
2. **Conformal prediction is mature.** Split-conformal calibration
   gives a measurable, finite-sample-valid uncertainty signal that
   LLM-only evals can't produce.
3. **The infra layer hasn't shipped.** Coding-RL companies (Codex,
   SWE-bench) own one shape; nobody owns scientific RL.
4. **Compliance demand.** EU AI Act + NIST AI RMF + ISO 42001 require
   *measured* model behaviour. Conformal-calibrated rewards are the
   right empirical primitive.

## What exists today (verifiable, no fake numbers)

- **10 environments** across compressed sensing, super-resolution,
  medical CT/MRI, phase retrieval. All live on Prime Intellect
  Environments Hub (verify: see README links).
- **464 automated tests** green, full suite under 10 s.
- **5+ frontier models benchmarked** (Claude Haiku 4.5, Sonnet 4.6,
  Opus 4.7, GPT-5.4, GPT-5.4-mini, GPT-5.4-nano). Per-row benchmark
  CSVs at `results/complete_matrix_{single,multi}_turn.csv` (90 + 132
  rows respectively).
- **Hosted FastAPI** (open, rate-limited 30 req/min/IP) deployed via
  Render / Fly.io / Docker IaC at [`deploy/api/`](../../deploy/api).
- **Python SDK** (`verifiable-labs` v0.1.0a1, sync + async) at
  [`packages/verifiable-labs/`](../../packages/verifiable-labs).
- **`verifiable` CLI** (this sprint): six subcommands (envs / run /
  compare / report / init-env / validate-env), zero new deps.
- **Hugging Face Spaces leaderboard** at `huggingface.co/spaces/stelioszach03/scientific-rl-benchmark`.
- **Paper** preprint (4 pages) at [`paper/main.pdf`](../../paper/main.pdf) — OpenReview submission pending.
- **Training-signal proof** — random search over a parameterised OMP
  on `sparse-fourier-recovery` finds a positive Δ on a held-out pool
  (Δ = +0.0024, 95 % CI [-0.001, +0.005] — CI includes zero, search
  budget too small for significance, but the loop is wired). See
  [`results/training_signal_demo.md`](../../results/training_signal_demo.md).

## Technical proof

- **Verifiable rewards.** Every env has a closed-form forward operator;
  `same seed + same git commit → same reward, bit-exact`.
- **Procedural regeneration.** Every test instance is generated from
  a seed; the seed pool is non-public and the calibration pool is
  outside the test-seed range. Effective instance count exceeds
  `1e15` per env (validated by `scripts/validate_env.py`).
- **Conformal calibration.** Per-env split-conformal quantile fitted
  on a held-back pool; empirical coverage on test seeds within ±5 pp
  of `1 - α` (α = 0.10). Regression-tested in
  `tests/test_calibration.py`.
- **Classical baselines.** Every env exposes `env.run_baseline(seed)`
  (OMP, FBP, HIO, bicubic, ZF-IFFT). LLM benchmarks gate against the
  baseline as the floor, never against zero.

## Why us

Solo founder: **Stelios Zacharioudakis** (NKUA Computer Science,
4th-year BSc). Head Engineer at AsklepiosMed (medical-AI startup,
480 doctors). Author of the v0.1 paper. ORCID `0009-0000-6021-5829`.
Independent execution of the entire v0.1 stack: 10 envs, hosted API,
SDK, CLI, leaderboard, paper.

## Wedge

Scientific RL environments where the answer is **physics-verifiable**
and **uncertainty-calibrated**. We start with 5 inverse-problem
families because they are continuous, ill-posed, and impossible to
solve by memorising static benchmark answers — they show up in:

- medical imaging (CT, MRI) — regulated, measurable
- compressed sensing — first-principles testbed
- physics inversion (phase retrieval, FWI) — high-stakes engineering

Once the wedge is anchored, the same primitives extend to: protein
structure (residue distogram), inverse rendering, electron
microscopy, geophysics, molecular dynamics.

## Expansion path

```
v0.1 alpha      research benchmark + 10 envs            ← shipped
v0.1.5 sprint   developer CLI + JSONL traces + reports  ← this commit
v0.2            auth / Redis sessions / 5 new envs /    ← Q3 2026
                first-class TRL+vLLM bindings /
                real-data variants for LoDoPaB+fastMRI
v0.3            community env marketplace / dashboard / ← Q4 2026
                attestation system / paid pilots
```

## Current limitations (honest)

- **No paying customers yet** (v0.1 is alpha, public endpoint is
  rate-limited and unauthenticated).
- **No SaaS billing**; `pip install` only.
- **No real RL integrations yet** — the env exposes the reward signal,
  but TRL / vLLM / GRPO bindings ship in v0.2.
- **Leaderboard is static** (HF Space backed by CSVs); v0.2 makes it
  interactive.
- **Single founder, no team yet.** YC + Neo Residency funding would
  enable hiring 1-2 ML / infra engineers to ship v0.2 in 90 days.

## Next 90 days

See [`ROADMAP_90_DAYS.md`](ROADMAP_90_DAYS.md). Headline:

- Week 1: public proof — CLI + demo video + preprint.
- Weeks 2-4: 3-5 design partners, dashboard, full RL training demo.
- Weeks 5-8: private env API for paying teams; CI integration.
- Weeks 9-12: paid design partners; v0.3 marketplace API.

## Ask

- Funding: pre-seed for the v0.2 + v0.3 build-out (auth, marketplace,
  RL training integrations, real-data variants).
- Distribution: introductions to AI post-training teams (Anthropic,
  OpenAI, Google, Meta, Mistral, xAI), scientific ML labs (medical
  imaging startups, robotics / physics agent teams), and AI-safety /
  evaluation teams (Apollo, METR).
- Pilots: 3-5 design-partner teams willing to provide a custom env
  and use the platform for 30 days. Free in v0.2; paid in v0.3.
