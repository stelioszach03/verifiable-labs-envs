"""Tiny smoke test: confirm OpenRouter API key works + measure tokens."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Read .env (no python-dotenv dep).
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import openai  # noqa: E402

key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
if not key:
    print("ERROR: no OPENROUTER_API_KEY in env", file=sys.stderr)
    sys.exit(2)

client = openai.OpenAI(api_key=key, base_url=base_url)

MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5-mini",
    "anthropic/claude-opus-4",
]

print(f"base_url: {base_url}")
print(f"key tail: ...{key[-6:]}")
print()
print("=== Probe each model with a 1-token completion ===")
for m in MODELS:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with one word."}],
            max_tokens=5,
            temperature=0.0,
        )
        usage = resp.usage
        out = resp.choices[0].message.content[:30] if resp.choices[0].message.content else ""
        print(f"  ✅ {m:<35}  out={out!r}  prompt_toks={usage.prompt_tokens}  "
              f"completion_toks={usage.completion_tokens}")
    except Exception as e:
        print(f"  ❌ {m:<35}  {type(e).__name__}: {str(e)[:120]}")
