# Roadmap

Public-facing summary of what's shipped, what's in flight, and what's
queued.

## Shipped — v0.1.0-alpha

- 10 environments across compressed-sensing, imaging (CT / MRI),
  physics inversion (phase retrieval), and 2D super-resolution
- Conformal-calibrated rewards (split-conformal at α = 0.10)
- Procedural regeneration verified at >1e15 effective instances per env
- 5-model paper-final benchmark: Claude (3 sizes), GPT-5.4 (3 sizes)
- Hugging Face Space leaderboard
- Hosted Evaluation API (FastAPI, 6 endpoints, 30 req/min/IP)
- Python SDK (`pip install verifiable-labs`, sync + async)
- Custom-env scaffold + validator (`scripts/create_env.py`,
  `scripts/validate_env.py`)
- Compliance report template + PDF generator
- This documentation site
- Training-proof notebook (prompt-search RLVR proxy)

## In flight — Tier 1 polish (next 2 weeks)

- Real domain `api.verifiable-labs.com` cutover (currently placeholder)
- Real domain `docs.verifiable-labs.com` cutover (currently GitHub
  Pages placeholder)
- PyPI publication of `verifiable-labs` (currently TestPyPI)
- Marketplace landing page on a real host (currently static HTML)
- Paper acceptance at OpenReview (under review)

## v0.2 — Tier 2 (Q3 2026)

The biggest increment. Targets:

- **Authentication.** Per-user API keys, dashboard, basic billing
  hooks.
- **Redis-backed sessions.** Sessions survive process restarts; multi-
  process API horizontal scaling becomes possible.
- **Server-side multi-turn dispatch.** The API runs `env.run_rollout`
  end-to-end instead of recording submissions for client-side replay.
- **Structured `answer` payloads.** Currently `answer_text: str`; v0.2
  accepts a `Prediction`-shaped dict directly, removing the JSON
  round-trip.
- **5 new envs.** Holographic 3D reconstruction, EM tomography,
  seismic FWI, inverse rendering, protein residue distogram. The
  first three are stubbed in `templates/inverse-problem/README.md`.
- **Real RL integration.** First-class `transformers` + `trl`
  bindings; per-step reward streaming.
- **Real-data variants.** Optional install of the LoDoPaB-CT and
  fastMRI public datasets behind data-use-agreement gates.

## v0.3 — Tier 3 (Q4 2026)

Speculative; subject to v0.2 outcomes.

- **Multi-domain visual env builder.** Drag-and-drop forward operator
  composer for new envs.
- **Real audit / SOC2 system.** Going beyond the
  capability-only compliance report into actual attestation.
- **Marketplace backend.** Today the marketplace landing page is
  vision-only; v0.3 may add a real submission flow for community
  envs with revenue share.
- **Scaled benchmark cadence.** Quarterly published benchmarks across
  all frontier models.

## What's *not* on the roadmap

- Conversational / chatbot evaluation. There are good benchmarks for
  that elsewhere (MT-Bench, ChatBot Arena, etc.).
- Open-ended generation evaluation. Same — out of scope.
- A model-hosting service. We score third-party models; we don't host
  them.

## Contributing

Open issues and pull requests welcome at
[github.com/stelioszach03/verifiable-labs-envs](https://github.com/stelioszach03/verifiable-labs-envs).
For new envs, follow [Tutorials → Creating a custom env](../tutorials/creating-custom-env.md);
the validator at `scripts/validate_env.py` is the merge gate.

For research collaborations or paid pilots, see [Contact](contact.md).
