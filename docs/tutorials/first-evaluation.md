# Tutorial: your first evaluation

End goal of this tutorial: score a frontier LLM's output on
`sparse-fourier-recovery` and read the components of the reward.
Estimated time: 5 minutes. No GPU, no training, no Python install
needed for the API path.

## Prerequisites

- An [OpenRouter](https://openrouter.ai) account (or any other
  chat-completions provider).
- `curl` + `jq` if you're using the API path; `pip` if you're using
  the SDK path.

## Step 1 — pick a model and an env

```bash
# List the 10 envs we ship.
curl https://api.verifiable-labs.com/v1/environments | jq '.environments[].id'
```

We'll use `stelioszach/sparse-fourier-recovery` because it's the
simplest single-turn env. Pick any frontier model — this tutorial
uses `anthropic/claude-haiku-4.5`.

## Step 2 — start a session

```bash
session=$(curl -s -X POST https://api.verifiable-labs.com/v1/sessions \
  -H "content-type: application/json" \
  -d '{"env_id": "stelioszach/sparse-fourier-recovery", "seed": 0,
       "env_kwargs": {"calibration_quantile": 2.0}}')

session_id=$(echo "$session" | jq -r .session_id)
prompt=$(echo "$session" | jq -r .observation.prompt_text)
system=$(echo "$session" | jq -r .observation.system_prompt)
```

`prompt_text` is what you send to the model as a user message;
`system_prompt` is the env's recommended system prompt.

## Step 3 — call the model

```bash
answer=$(curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "content-type: application/json" \
  -d "$(jq -n --arg sys "$system" --arg user "$prompt" '{
    model: "anthropic/claude-haiku-4.5",
    temperature: 0,
    messages: [
      {role: "system", content: $sys},
      {role: "user",   content: $user}
    ]
  }')" | jq -r '.choices[0].message.content')
```

The model emits JSON like:

```json
{"support_idx": [12, 47, 91, …], "support_amp_x1000": [800, -1200, …]}
```

## Step 4 — submit and read the score

```bash
curl -s -X POST "https://api.verifiable-labs.com/v1/sessions/$session_id/submit" \
  -H "content-type: application/json" \
  -d "$(jq -n --arg ans "$answer" '{answer_text: $ans}')" | jq
```

Output:

```json
{
  "session_id": "s-9a3b4c…",
  "reward": 0.351,
  "components": {
    "nmse": 0.49,
    "support": 0.20,
    "conformal": 0.40
  },
  "coverage": 0.40,
  "parse_ok": true,
  "complete": true,
  "meta": {"weights": {"nmse": 0.4, "support": 0.3, "conformal": 0.3}}
}
```

## Reading the score

- `reward = 0.351` is the weighted bundle of the three components.
- `components.nmse = 0.49` — moderate point-estimate fidelity.
- `components.support = 0.20` — only 2 of 10 support indices correct.
- `components.conformal = 0.40` — the model's stated uncertainty
  intervals contained the truth on 40 % of coordinates, well below
  the 90 % target. The model is *over-confident*.

The `coverage` field is the empirical conformal coverage on this seed
(here `0.40`). For a well-calibrated model on this env we'd expect
~0.90 — the conformal-score component penalises the gap.

## Doing this from Python

The same flow with the SDK:

```python
from verifiable_labs import Client
import os
import openai

c = Client(base_url="https://api.verifiable-labs.com")
env = c.env("stelioszach/sparse-fourier-recovery")
sess = env.start_session(seed=0, env_kwargs={"calibration_quantile": 2.0})

# Call the model with the env-supplied prompts.
oai = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1")
resp = oai.chat.completions.create(
    model="anthropic/claude-haiku-4.5",
    temperature=0,
    messages=[
        {"role": "system", "content": sess.observation.system_prompt},
        {"role": "user",   "content": sess.observation.prompt_text},
    ],
)
sess.submit(resp.choices[0].message.content)
print(f"reward={sess.history[-1].reward:.3f}  components={sess.history[-1].components}")
```

## Next

- Run this on multiple seeds and a few models to populate a small
  leaderboard locally.
- Try `sparse-fourier-recovery-multiturn` — the env supplies
  follow-up turns with residual feedback. The SDK's `Session` class
  handles the multi-turn loop transparently.
- See [Compliance reports](compliance-reports.md) once you have a
  benchmark CSV with multiple models scored.
