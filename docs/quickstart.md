# Quickstart

Three on-ramps depending on what you need. Each takes under 5 minutes.

## A. Hosted REST API — no Python install

The fastest way to score one model output against one env.

```bash
# 1. Health check.
curl https://api.verifiable-labs.com/v1/health

# 2. List envs.
curl https://api.verifiable-labs.com/v1/environments | jq '.environments[].id'

# 3. Start a session.
session_id=$(curl -s -X POST https://api.verifiable-labs.com/v1/sessions \
  -H "content-type: application/json" \
  -d '{"env_id": "stelioszach/sparse-fourier-recovery", "seed": 0,
       "env_kwargs": {"calibration_quantile": 2.0}}' \
  | jq -r .session_id)

# 4. Submit your model's output.
curl -X POST "https://api.verifiable-labs.com/v1/sessions/$session_id/submit" \
  -H "content-type: application/json" \
  -d '{"answer_text": "{\"support_idx\": [12, 47, 91], \"support_amp_x1000\": [800, -1200, 450]}"}'

# 5. Inspect the full session state (history of submissions).
curl https://api.verifiable-labs.com/v1/sessions/$session_id
```

Full schema lives in the OpenAPI UI at `https://api.verifiable-labs.com/docs`.

!!! warning "Alpha rate limit"
    The public endpoint is open (no auth) but rate-limited to 30 req/min/IP.
    See [REST API reference](api-reference/rest-api.md) for the v0.2 auth
    plan.

## B. Python SDK — for scripted evaluations

```bash
pip install verifiable-labs
```

```python
from verifiable_labs import Client

with Client(base_url="https://api.verifiable-labs.com") as c:
    # 1. Health.
    print(c.health())

    # 2. Drive a session.
    env = c.env("stelioszach/sparse-fourier-recovery")
    result = env.evaluate(
        seed=42,
        answer='{"support_idx": [12, 47, 91], "support_amp_x1000": [800, -1200, 450]}',
        env_kwargs={"calibration_quantile": 2.0},
    )
    print(f"reward={result.reward:.3f} parse_ok={result.parse_ok}")
    print(f"components={result.components}")

    # 3. Multi-turn loop.
    session = env.start_session(seed=42)
    for turn in range(3):
        observation = session.observation if turn == 0 else session.history[-1].response
        # … your model produces `answer` from observation …
        session.submit(answer)
        if session.complete:
            break
    print(f"final reward={session.history[-1].reward:.3f}")

    # 4. Leaderboard for cross-model comparison.
    lb = c.leaderboard("sparse-fourier-recovery")
    for row in lb.top_models(n=5):
        print(f"{row.model:30s} mean={row.mean_reward:.3f} parse_fail={row.parse_fail_rate:.0%}")
```

`AsyncClient` mirrors the sync API one-to-one with `await`.

## C. Local envs — for training loops + custom code

```bash
git clone https://github.com/stelioszach03/verifiable-labs-envs.git
cd verifiable-labs-envs
pip install -e ".[dev]"
```

```python
from verifiable_labs_envs import load_environment
from verifiable_labs_envs.solvers import OpenRouterSolver

env = load_environment("sparse-fourier-recovery-multiturn")
solver = OpenRouterSolver(model="anthropic/claude-haiku-4.5")

instance = env.generate_instance(seed=0)
result = env.run_rollout(solver, instance)
print(f"reward={result['reward']:.3f}")
```

Local mode gives you direct access to:

- `env.generate_instance(seed)` — deterministic problem instance from a seed
- `env.score(prediction, instance)` — bit-exact reward from a typed
  `Prediction` dataclass
- `env.run_rollout(solver, instance)` — full multi-turn dispatch with
  any solver implementing the `LLMSolver` interface

This is the surface used by the [training-proof notebook](tutorials/training-with-envs.md).

## Next

- [REST API reference](api-reference/rest-api.md) — every endpoint, every field
- [Tutorials](tutorials/first-evaluation.md) — guided walk-throughs
- [Concepts](concepts/conformal-rewards.md) — why the rewards are calibrated
