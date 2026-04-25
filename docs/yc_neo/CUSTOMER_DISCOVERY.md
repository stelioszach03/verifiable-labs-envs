# Customer discovery

30 target profiles across 7 segments, plus the outreach scripts and
discovery-call template the founder will run during weeks 2-4 of the
[`ROADMAP_90_DAYS.md`](ROADMAP_90_DAYS.md).

## Segments + target profiles

### 1. AI post-training teams (frontier labs)

The teams that build the RL training loops for frontier LLMs.

1. Anthropic — alignment / RLHF team
2. OpenAI — post-training research
3. Google DeepMind — Gemini post-training
4. Meta — LLaMA post-training
5. Mistral — RL fine-tuning team
6. xAI — Grok post-training

**Why they care:** Need verifiable RL environments for science /
math / engineering tasks. Today's options are ad-hoc.
**Buying motion:** evaluation contract, then deeper integration.

### 2. Scientific ML labs (academic / corporate)

University labs and corporate research labs publishing on RL +
science.

7. Stanford CRFM (Foundation Model Research)
8. MIT CSAIL — RL groups
9. Princeton — Princeton Language and Intelligence
10. CMU — Multi-modal Intelligence
11. ETH Zurich — Computational Inverse Problems
12. Allen Institute for AI (AI2)

**Why they care:** Need reproducible benchmarks for papers; want
calibrated uncertainty signals.
**Buying motion:** free academic tier; co-authored case studies.

### 3. Medical imaging AI startups

13. PathAI — pathology AI
14. Aidoc — radiology workflows
15. Rad AI — radiology reporting
16. Iterative Health — gastroenterology
17. AsklepiosMed (founder's day job, internal pilot only — does not
    count as an external customer in pitch material)

**Why they care:** Need to evidence model capability + uncertainty
to FDA / CE-mark reviewers. Conformal coverage is the right
artefact.
**Buying motion:** compliance report subscription; per-model
evaluation.

### 4. Robotics / physics / chemistry agent teams

18. Physical Intelligence (Pi) — robotics RL
19. Boston Dynamics AI Institute
20. Periodic Labs — chemistry agents
21. NVIDIA Isaac Sim / robotics teams
22. DeepMind Robotics

**Why they care:** Inverse problems show up in vision and control;
classical baselines (Kalman filters, OMP, FBP) are everywhere.
**Buying motion:** custom env contract for their domain.

### 5. Regulated AI teams (finance / health / energy)

23. JPMorgan AI Research — risk modelling
24. Schmidt Sciences AI Safety Institute
25. Lawrence Livermore National Lab — physics modelling
26. Recursion / Insitro — biology RL

**Why they care:** Compliance-grade evaluation evidence; the
[`templates/compliance-report/`](../../templates/compliance-report)
artefact is the wedge.
**Buying motion:** annual evaluation subscription.

### 6. AI safety / evaluation teams

27. METR — model evaluation
28. Apollo Research — eval engineering
29. UK AI Safety Institute
30. Redwood Research — interpretability

**Why they care:** Verifiable, reproducible environments where they
can measure capability progress over time.
**Buying motion:** research collaboration; later, evaluation
contract.

### 7. AI infrastructure founders (peers)

Not direct customers, but distribution partners.

(No specific names listed; we'll meet these at YC / Neo Residency
intros and through the AI infra meetup circuit.)

**Why they matter:** Co-marketing, technical-mentor relationships,
and integration partnerships (Modal, Together AI, Replicate, etc.).

## Outreach questions (discovery)

Before pitching anything, we need to know:

1. **Are you currently doing RL training on scientific / continuous
   tasks?** If yes — what reward function? If no — when do you
   expect to?
2. **What's your current evaluation harness?** LM Eval Harness, a
   homegrown fork, or a one-off pipeline per project?
3. **Do you measure calibrated uncertainty in your model outputs?**
   If yes, how. If no, would it change downstream decisions?
4. **What's your worst recent failure mode** in a model that
   passed evals but failed in production / regulatory review?
5. **Who owns "model evaluation" in your org?** A research team, an
   infra team, or someone wearing both hats?
6. **What's your current annual spend** on third-party evaluation
   tooling (LM Eval Harness, OpenAI Evals, custom builds)?

These six questions take 20 minutes on a discovery call and rule
out 80 % of prospects.

## Outreach templates

### Cold email (post-training team)

```
Subject: Verifiable RL environments for scientific reasoning (Anthropic post-training)

Hi <name>,

I'm Stelios — I build verifiable RL environments for scientific
reasoning. We have 10 envs live (compressed sensing, CT, MRI, phase
retrieval, super-resolution) with conformal-calibrated rewards and
procedural problem regeneration — physics-verifiable, impossible to
solve by memorising static benchmark answers.

Today we ship a CLI, SDK, and hosted API: an engineer can run
`verifiable run --env <id> --agent <yours> --n 30 --out runs/me.jsonl`
and get a Markdown evaluation report in under 60 seconds.

Would 20 minutes work for a discovery call? I want to understand
what's blocking your team's scientific-RL evaluation today and
whether a calibrated reward signal would be useful.

Repo: https://github.com/stelioszach03/verifiable-labs-envs
Paper: https://github.com/stelioszach03/verifiable-labs-envs/blob/main/paper/main.pdf

— Stelios
```

### Cold email (scientific ML lab)

```
Subject: Procedurally-regenerated scientific RL benchmark — collab?

Hi <name>,

I'm Stelios. I just shipped v0.1 of a benchmark / training-signal
platform for scientific RL: 10 inverse-problem environments
(sparse Fourier, CT, MRI, phase retrieval, super-resolution) with
conformal-calibrated rewards. Preprint at <link>.

The benchmark is built to be reproducible and contamination-resistant
(every test instance procedurally regenerated, ~`1e15` effective
instance count per env). Curious if your group would be interested
in either:

(a) Running your model on the benchmark for a co-authored
    short-paper write-up.
(b) Contributing a custom env in your domain via our scaffold
    (`verifiable init-env`) — could be useful for your future
    benchmark releases.

20-min call?

— Stelios
```

### Cold email (medical imaging startup)

```
Subject: Compliance-grade model evaluation reports for medical imaging AI

Hi <name>,

You're probably aware EU AI Act + FDA SaMD reviewers are increasingly
asking for *measured* model capability and *measured* uncertainty,
not just accuracy claims.

I've built a tool that produces compliance-grade evaluation reports
from your model: per-env capability table, parse-fail rate,
empirical conformal coverage vs target, recommendations. Markdown
+ PDF outputs. Reproducible from a CSV your team owns.

10 envs live today (sparse Fourier, CT, MRI, super-resolution, phase
retrieval). v0.1 is alpha; we're looking for 3-5 design partners to
shape v0.2.

20 minutes to walk through it?

— Stelios
```

## Discovery-call script (20 min)

**0-2 min — context.** Who I am, what Verifiable Labs is in 30
seconds, what I want to learn. (Not what I want to sell.)

**2-5 min — their world.** "What's your current evaluation
harness?" Listen. Don't pitch.

**5-10 min — pain.** "What's your worst recent eval failure?"
"What's the most expensive thing you've measured wrong?"

**10-15 min — capability narrowing.** Match what we ship to a
specific pain they named. If no match — "this isn't right for you,
who do you know who'd care?"

**15-18 min — concrete next step.** "If I send you a 5-minute
walkthrough, would you watch it?" or "Could I add you to the design-
partner waitlist?" — small, low-commitment.

**18-20 min — close.** Specific date for follow-up. Specific
artefact (video, doc, demo run on their model). No vague "let's
keep in touch."

## Tracking

Outreach log lives in a private spreadsheet (founder owns) — name,
segment, email date, response, next step, notes. Not committed to
this repo.

## When to walk away

- They want a free model-comparison report and won't engage past
  that. (We are not a free LLM eval service.)
- They want production guarantees we don't ship. (v0.1 is alpha;
  saying so is the right move.)
- They expect us to integrate with their proprietary stack before
  we have paying customers. (We will do this for paid pilots, not
  for free.)
