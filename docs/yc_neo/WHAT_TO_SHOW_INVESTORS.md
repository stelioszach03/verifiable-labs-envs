# What to show investors

90-second structure for a YC + Neo Residency demo recording, broken
down by what's on screen and what comes out of your mouth. See
[`DEMO_COMMANDS.md`](DEMO_COMMANDS.md) for the exact terminal
commands.

## 0:00 – 0:10 — hook (10 s)

**On screen:** `README.md`'s first paragraph in a code editor or
GitHub view.

**Say:**

> "Verifiable Labs is the API, SDK, and CLI layer for evaluating
> and training scientific AI agents on verifiable RL environments."

That's the entire hook. Don't elaborate. Cut.

## 0:10 – 0:40 — demo (30 s)

**On screen:** terminal at full width, one `verifiable` command
per beat. Use the exact commands in
[`DEMO_COMMANDS.md`](DEMO_COMMANDS.md).

**Beats (each ~5 s):**

1. `verifiable envs` → "10 envs across 5 scientific domains."
2. `verifiable run --env sparse-fourier-recovery --agent
   examples/agents/zero_agent.py --n 3 --out runs/demo.jsonl` →
   "Three episodes, zero API cost. JSONL trace per episode."
3. `verifiable report --run runs/demo.jsonl --out reports/demo.md`
   → "12-section Markdown evaluation report."
4. `cat reports/demo.md | head -30` → silent — let the viewer read.

## 0:40 – 1:10 — differentiation (30 s)

**On screen:** cut to the README's "Why verifiable" section, or to
the leaderboard tab on HF Spaces.

**Say:**

> "Most AI eval tools test chatbots and apps. Verifiable Labs
> generates scientific environments with **objective rewards**,
> **calibrated uncertainty**, **procedural regeneration**,
> **classical baselines**, and **training-signal potential** —
> tasks that are continuous, uncertainty-sensitive, and impossible
> to solve by memorising static benchmark answers."

*(Read the bolded list slowly; those are the five pillars.)*

> "We start with CT, MRI, compressed sensing, phase retrieval, and
> super-resolution because they are the cleanest test cases for
> measurable scientific reasoning. The same primitives extend to
> protein structure, inverse rendering, electron microscopy, and
> molecular dynamics."

## 1:10 – 1:30 — technical proof (20 s)

**On screen:** the meta-benchmark v3 summary, then the
`results/training_signal_demo.md` headline, then the paper PDF cover.

**Say:**

> "Today we have 10 envs live, 5+ frontier models benchmarked,
> 464 automated tests green, and a training-signal proof that
> shows the env reward function can drive parameter optimisation.
> The methodology paper is a preprint pending OpenReview review.
> All numbers are reproducible from the repo."

**Don't claim:**

- ❌ "We have customers" (we don't yet — we're pre-seed alpha)
- ❌ "We've raised X" (no funding yet)
- ❌ "Frontier providers use us" (they don't — yet)

## 1:30 – 1:40 — ask (10 s)

**On screen:** founder face on camera, repo URL on screen.

**Say:**

> "We're raising a pre-seed for v0.2: auth, marketplace API, and
> first-class TRL / vLLM bindings. Looking for intros to AI
> post-training teams, scientific ML labs, and AI-safety
> evaluators. Repo + paper at the link below."

End on the URL. Don't pad.

## What to optimise for

1. **Speed of the live demo.** Practice until you can hit the four
   commands in 25 s without backtracking.
2. **Reading rhythm of the differentiation paragraph.** The five
   bolded primitives need to land separately.
3. **Honest scope.** Every claim above is verifiable in the repo or
   the CSV files. Nothing is fabricated. Investors notice when
   numbers shift across recordings — pin them.
4. **Cut the academic detail.** The paper exists; link to it. Don't
   read from it.

## What to send after the demo

If the recording lands well, send the demo URL plus three artefacts:

1. [`ONE_PAGER.md`](ONE_PAGER.md) — read-in-90-seconds business case
2. [`COMPETITIVE_POSITIONING.md`](COMPETITIVE_POSITIONING.md) —
   honest comparisons
3. [`ROADMAP_90_DAYS.md`](ROADMAP_90_DAYS.md) — what we ship next

Skip everything else from `docs/yc_neo/` unless asked. The
`CUSTOMER_DISCOVERY.md` file is internal-only; don't share it
externally.
