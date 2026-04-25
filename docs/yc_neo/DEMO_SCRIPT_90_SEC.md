# 90-second demo script

Practise this verbatim. Reads naturally; total runtime is the user's
hands-on-keyboard time, not the slides.

## 0:00 – 0:10 — problem (10 s)

> "Frontier models are trained with verifiable rewards. But today's
> RL environments are mostly text-only, saturate quickly, and miss
> the continuous, uncertainty-sensitive reasoning that real science
> requires. There's no shared infrastructure for evaluating and
> training scientific AI agents."

## 0:10 – 0:25 — solution (15 s)

> "Verifiable Labs is the API, SDK, and CLI layer for that. We
> generate scientific RL environments with objective rewards,
> calibrated uncertainty, procedural regeneration, and classical
> baselines — physics-verifiable problems impossible to solve by
> memorising static benchmark answers."

## 0:25 – 0:55 — CLI / API / SDK demo (30 s)

Open a terminal. Run these in order. Each command outputs in <2 s.

```bash
# 1. Install (already done; for the recording assume the env is set up).
pip install -e ".[dev]"     # → verifiable command available

# 2. List the 10 envs.
verifiable envs

# 3. Run a zero-amplitude agent on sparse-Fourier (3 episodes).
verifiable run \
    --env sparse-fourier-recovery \
    --agent examples/agents/zero_agent.py \
    --n 3 --out runs/demo.jsonl

# 4. Render a 12-section Markdown evaluation report.
verifiable report --run runs/demo.jsonl --out reports/demo.md

# 5. (Show the report opening in any reader.)
open reports/demo.md       # or `cat reports/demo.md`
```

Talking over it:

> "I'm running a dummy agent — zero amplitudes — for 3 episodes on
> the sparse-Fourier env. The CLI writes a per-episode JSONL with a
> stable schema, then `verifiable report` turns it into a 12-section
> Markdown. Same flow works for an OpenAI agent, a subprocess in any
> language, or the hosted REST API."

## 0:55 – 1:15 — benchmark + differentiation (20 s)

Cut to the leaderboard tab.

> "Today, on 10 envs across compressed sensing, CT, MRI, super-
> resolution, and phase retrieval, no frontier LLM beats the
> classical baseline on every env. Claude Haiku 4.5 leads at 0.604
> cross-env mean. The benchmark is reproducible from `results/`
> CSVs; the paper's preprint is in `paper/main.pdf`.

> Most AI eval tools test chatbots and apps. Verifiable Labs
> generates scientific environments with calibrated uncertainty,
> procedural regeneration, and training-signal potential — that's
> the wedge."

## 1:15 – 1:30 — why now / ask (15 s)

> "RLVR is going mainstream this year. Coding-RL is owned by
> OpenAI / Anthropic / SWE-bench. Scientific-RL is the unmet wedge,
> and the compliance demand from EU AI Act and NIST AI RMF wants
> measured, reproducible model behaviour — exactly what we ship.

> We're raising a pre-seed to hire 1-2 engineers for v0.2 (auth,
> marketplace, RL training bindings) and ship 5 new envs over the
> next 90 days. Looking for intros to AI post-training teams,
> scientific ML labs, and AI-safety evaluators."

---

## Recording tips

- **Pre-set the environment.** Run the install, the API key export,
  and `mkdir -p runs reports` *before* hitting record. The demo
  starts from `verifiable envs`.
- **Resize the terminal** to about 100×30 — wide enough for the
  per-episode log lines, narrow enough that one screen captures
  it without scrolling.
- **Pre-load the leaderboard tab** in another window. `Cmd-Tab` for
  the cut at 0:55.
- **Don't read the report aloud.** Open it for visual proof, then
  cut. The viewer reads it; you don't.
- **Time the run-step deliberately.** `verifiable run` takes ~1 s
  for n=3; the per-episode log lines auto-scroll. Don't cut that
  output — it's the most cinematic part of the loop.
- **Practice the close.** "Looking for intros to X, Y, Z. Funding
  for v0.2 builds out auth + marketplace + RL bindings." That's the
  ask; it doesn't change.
