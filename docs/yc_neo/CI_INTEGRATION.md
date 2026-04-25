# CI integration

How to wire `verifiable run` into a GitHub Actions workflow that
gates merges on the platform's reward signal.

## TL;DR

Copy [`.github/workflows/verifiable-eval-example.yml`](../../.github/workflows/verifiable-eval-example.yml)
into your own repo. It runs the bundled zero agent on
`sparse-fourier-recovery` (3 episodes, no API key required) and
uploads JSONL + Markdown as workflow artefacts.

## Replacing the agent

The example workflow uses `examples/agents/zero_agent.py`. To run
*your* agent in CI:

1. Commit your agent file (`my_agent.py`) to your repo.
2. Edit the workflow's `Run zero agent` step:

   ```yaml
   env:
     VL_AGENT_PATH: ./my_agent.py
   ```

3. (Optional) For an OpenAI-compatible HTTP agent, set the API key as
   a repo secret and pass it through:

   ```yaml
   env:
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
     VL_AGENT_PATH: "openai:anthropic/claude-haiku-4.5"
   ```

   Then the agent spec is `openai:<model>` (no Python file needed).

## Gating merges on a reward floor

The example's last step is a *soft* check (warns, doesn't fail). To
make it a *hard* gate, change `::warning::` to a non-zero exit:

```python
if mean < threshold:
    print(f"::error::mean reward {mean:.3f} below floor {threshold:.3f}")
    sys.exit(1)
```

Pick the threshold carefully:

- For `sparse-fourier-recovery` with the zero agent the floor is
  ~0.34. Anything **below** that means the JSON parser broke.
- For your own agent, run a 30-episode baseline locally first,
  then set the threshold ~1 std below the mean.

## Tightening the trigger

The example runs on push to `main` or `funding-sprint-yc-neo`. For
your repo, common tighter triggers:

```yaml
on:
  pull_request:
    paths:
      - "src/my_model/**"
      - "my_agent.py"
  workflow_dispatch:
```

Combined with branch-protection rules on `main`, this gates every
PR on the reward signal.

## Artefacts

The workflow uploads:

- `runs/ci_demo.jsonl` — per-episode trace, schema-stable, readable
  by `verifiable report` / `verifiable compare`
- `reports/ci_demo.md` — 12-section Markdown report

Available from the Actions tab → run → "verifiable-eval-demo"
artefact bundle.

## Cost

The bundled `zero_agent.py` makes **no network calls**. The full
workflow runs in ~30-60 seconds on a `ubuntu-latest` runner — well
within the GitHub free tier.

If your agent calls a paid API (OpenAI / OpenRouter / Anthropic),
the cost is the cost of N episodes × your token budget per
episode. Cap it with `--n` and (eventually) per-key budget caps in
v0.2 of the Verifiable Labs API.

## Limitations of the example workflow

- **One env, one agent.** For multi-env / multi-agent matrices,
  use a `matrix:` strategy and aggregate the JSONL results in a
  follow-up step (the trace schema is stable; concatenating
  multiple runs is `cat`).
- **Markdown report only.** PDF rendering needs `pandoc` (install
  via `apt-get install -y pandoc` in the workflow); Markdown is the
  default because it doesn't need external deps.
- **No reward-history baseline.** The workflow scores a single PR,
  not "this PR vs the last 30 days". v0.2 adds a hosted history
  service so workflows can do regression-style comparison.

## Beyond the example

For more advanced patterns — multiple agents, custom env, scheduled
benchmarks — see [`docs/api-reference/cli.md`](../api-reference/cli.md)
and the per-subcommand `--help`.
