# Competitive positioning

Honest comparison against the categories Verifiable Labs touches.
Not exhaustive; covers the references YC and Neo Residency reviewers
will recognise.

## TL;DR

> Most AI eval tools test chatbots and apps. Verifiable Labs
> generates **scientific** environments with **objective rewards**,
> **calibrated uncertainty**, **procedural regeneration**, **classical
> baselines**, and **training-signal potential**.

The differentiation is not "better than X" on a single axis. It's
**owning a coherent stack** (scientific domains × verifiable rewards
× calibrated uncertainty × developer ergonomics) that no single
existing tool covers.

## Generic LLM eval tools

| | LM Eval Harness | Ragas | OpenAI Evals | Verifiable Labs |
|---|---|---|---|---|
| **shape** | static answer-key benchmarks | RAG-pipeline evaluation | LLM judge / heuristic eval | procedural scientific RL envs |
| **reward** | exact match / multiple choice | RAG-quality scores | LLM judge / regex / function | physics-verifiable + conformal |
| **regenerable instances** | no | n/a | no | yes (~`1e15`+ per env) |
| **uncertainty calibration** | no | no | no | split-conformal, target ±5 pp |
| **classical baseline** | n/a | n/a | n/a | per-env (OMP, FBP, HIO, …) |
| **training-signal** | no | no | partial | yes (verifiable reward) |

These tools are excellent for chatbot / RAG / general-purpose LLM
quality. They're not built for continuous scientific reasoning, and
they don't expose a reward signal you can use as an RL training
signal — that's a different shape of problem.

## Agent benchmark tools

| | SWE-bench | OSWorld / WebArena | TAU-Bench / AppWorld | Verifiable Labs |
|---|---|---|---|---|
| **domain** | software engineering | web / desktop agents | tool-use agents | scientific RL |
| **reward** | unit-test pass rate | task completion | tool-call correctness | conformal-calibrated reward |
| **regenerable** | no (frozen issues) | partial (state resets) | yes | yes |
| **classical baseline** | none (LLMs vs LLMs) | none | none | yes (OMP, FBP, HIO, …) |
| **uncertainty** | binary pass/fail | task success | tool dispatch correctness | calibrated |

These benchmarks own different verticals. They share with us the
"verifiable reward" property but not the scientific-domain wedge or
the conformal-calibration primitive. There's no overlap on customers
in the near term.

## Sandbox / runtime infrastructure

| | E2B / Modal / Daytona | Inspector / browser MCP | Verifiable Labs |
|---|---|---|---|
| **shape** | code execution sandbox | live browser instrumentation | scientific RL envs + reward |
| **what they own** | secure compute substrate | real-world action surface | reproducible reward function |
| **overlap with us** | none — we run on top of compute | none — we don't drive UIs | — |

Sandbox and instrumentation infra is *complementary*: a research
team running RLVR on Verifiable Labs envs might use Modal for
compute and E2B for tool sandboxing. We don't compete; we layer.

## Runtime guardrails

| | Lakera / Robust Intelligence / Lasso | Verifiable Labs |
|---|---|---|
| **mode** | inline guard at inference time | offline / batch evaluation |
| **goal** | block jailbreaks, prompt injection, sensitive content | quantify capability + uncertainty |
| **artefact** | "yes / no, allow this prompt" | "reward + components + coverage + report" |

Different problem shape entirely. Guardrail vendors block bad inputs
in production; we evaluate model behaviour against verifiable
ground truth before deployment. A buyer might use both.

## Coding-RL infrastructure

| | Replit Agent / Devin / Codex / SWE-RL | Verifiable Labs |
|---|---|---|
| **domain** | software engineering tasks | scientific reasoning tasks |
| **reward** | unit-test pass / code execution | physics + conformal |
| **regenerability** | partial (issue-bank backed) | yes, by construction |
| **calibration** | no | yes (split-conformal) |

These are the closest spiritual neighbours: dedicated RL training
infra for one vertical. We are explicitly not trying to compete with
them on coding — we go deeper on a different vertical (scientific +
medical + physics inversion) where the reward shape they use
(unit tests) doesn't apply.

## Why we don't fit any existing bucket cleanly

A scientific RL environment platform needs to be all of:

1. a **benchmark** (so customers can compare models),
2. a **training signal source** (so customers can fine-tune models),
3. a **developer surface** (CLI, SDK, API — so a single engineer can
   adopt it without a service desk), and
4. an **uncertainty primitive** (because in regulated domains,
   calibrated bounds are non-negotiable).

Each existing competitor owns 1-2 of those. None owns all four. That
gap is our position.

## Uncomfortable honest takes

- **We are pre-revenue, pre-customer.** "Position" doesn't mean
  market position yet; it means architectural position. Customers
  arrive in v0.2.
- **Coding-RL ate the first wave.** Scientific RL is small now; the
  bet is that medical / physics / biology RLVR catches up over the
  next 12-24 months.
- **Conformal calibration is undervalued today.** Most teams skip
  it. Compliance demand is growing slowly. We are betting on the
  trend curve, not the current-state market.
- **A frontier-model provider could replicate us.** They probably
  won't, because scientific environments are not where they extract
  value. But it's a real risk, and our defence is shipping faster
  than they can decide to allocate the team.

## Defence

1. **Domain expertise.** Founder authored the v0.1 paper; the
   methodology document is the canonical reference. Replicating
   "10 envs with the right reward function" takes >3 months.
2. **Calibration discipline.** Conformal calibration is easy to
   add wrong. Our regression tests catch ±5 pp drift; competitors
   shipping without that infrastructure will be obvious.
3. **Developer surface.** CLI + SDK + API + JSONL traces + reports
   on day one. A team trying to bolt this onto an existing
   benchmark adds months of UX work.
4. **Compliance-grade artefacts.** Every run produces a JSONL trace
   that an auditor can verify. That's a contract advantage, not just
   a feature.
