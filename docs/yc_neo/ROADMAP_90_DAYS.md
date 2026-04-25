# 90-day roadmap

Funded scope: pre-seed support enables 1-2 ML / infra hires. The
solo-founder version of this plan delivers the same milestones but
on a longer cycle (~6 months vs 3).

## Week 1 — public proof

**Goal:** make it impossible for an interested ML engineer to miss us.

- [x] CLI / SDK / API shipped end-to-end (this sprint).
- [x] `verifiable run → report` works in <60 s from a clone.
- [ ] Demo video recorded (uses [`DEMO_SCRIPT_90_SEC.md`](DEMO_SCRIPT_90_SEC.md)).
- [ ] Preprint pushed to OpenReview / arXiv with a working
      reproducibility link.
- [ ] HF Spaces leaderboard refreshed with the v3 meta-benchmark.
- [ ] PyPI publish of `verifiable-labs` (currently TestPyPI).
- [ ] Show HN + r/MachineLearning post the day the video lands.

## Weeks 2-4 — design partners

**Goal:** 3-5 teams agreeing to use the platform on their own model
for 30 days, free, in exchange for a per-week feedback call.

- [ ] Outreach to 30 target profiles ([`CUSTOMER_DISCOVERY.md`](CUSTOMER_DISCOVERY.md)).
- [ ] Calibration validator hardened to run in <5 s on any new env.
- [ ] Static dashboard at `dashboard.verifiable-labs.com` (read-only;
      same source CSVs as the HF Space).
- [ ] **Training demo upgraded:** swap the random-search demo for a
      DSPy / TRL-PPO loop on `sparse-fourier-recovery-multiturn`,
      target a positive held-out Δ with statistical significance.
- [ ] Drop `pip install verifiable-labs[envs]` extras so the SDK
      bundle includes the env stack on demand (closes the
      "different package name" UX gap).

## Weeks 5-8 — private API + CI integration

**Goal:** the platform becomes part of a customer's commit gate.

- [ ] **Auth + tenant model.** Per-customer API keys, per-key rate
      limits, per-key budget caps. (v0.2 scope.)
- [ ] **Redis-backed sessions.** Sessions survive restart;
      multi-process API horizontal scale.
- [ ] **Server-side multi-turn dispatch.** API runs `env.run_rollout`
      end-to-end; SDK no longer assembles follow-up turns
      client-side.
- [ ] **GitHub Action templates.** `verifiable-eval-example.yml` (this
      sprint) is the floor; ship 3 more (cost-gated, reward-gated,
      coverage-gated) for different customer use cases.
- [ ] **5 new envs.** Holographic 3D, EM tomography, seismic FWI 1D,
      inverse rendering, protein residue distogram. Each follows the
      `templates/inverse-problem/` scaffold; merge gate is
      `verifiable validate-env` green.

## Weeks 9-12 — paid design partners + marketplace API

**Goal:** $-revenue or signed LOI from at least one design partner.

- [ ] Convert 1-2 of the unpaid design partners to paid pilots.
      Pricing: $5k/mo for "evaluate one model on the platform" tier
      (rough — anchored on what the customer pays for compute).
- [ ] **Custom env builder API.** Customers submit private envs
      through the same scaffold, but via the API instead of a PR.
      Each customer's envs live in their tenant; cross-tenant
      privacy enforced.
- [ ] **First-class TRL bindings.** `pip install verifiable-labs[trl]`
      exposes a `VerifiableLabsRewardModel` you can wire into a TRL
      `PPOTrainer`. Same interface for `vllm` rollouts.
- [ ] **Compliance report v2.** PDF generator (Tier-1) gains
      attestation-grade artefacts: cryptographic commit hash,
      benchmark CSV digest, reproducibility instructions inside the
      PDF body. Useful for AI Act / NIST AI RMF exhibits.

## What's *not* in 90 days

Scoped out so the plan is honest:

- **Not v0.3 marketplace backend.** Real environment marketplace with
  community contributions, revenue share, payouts — Q4 2026 at
  earliest, depends on v0.2 outcomes.
- **No new domain (audio, NLP, code).** We anchor in scientific
  inverse problems for v0.2; expansion is v0.3.
- **No model hosting service.** We score models; we don't host them.
- **No regulatory attestation.** We ship the *artefacts*; the legal
  attestation belongs to the customer's compliance team.

## Hiring plan (if funded)

1. **Senior ML engineer** (RL training infra) — TRL / vLLM bindings,
   real RL training loop, env tuning for reward-density.
2. **Infra / platform engineer** — auth, Redis, multi-tenant API,
   GitHub Action library, dashboard.

Solo founder remains responsible for the science (envs, reward
functions, calibration), the public surface (paper, demo, sales),
and the platform architecture.

## Tracking

This file is the source of truth. Updated weekly with checkbox
progress. Commit history on `main` shows what shipped vs what
slipped.
