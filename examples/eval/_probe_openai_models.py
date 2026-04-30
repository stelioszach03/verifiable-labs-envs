"""Probe OpenAI-family models on OpenRouter to find a working alternative."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
for line in ENV_PATH.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

import openai

client = openai.OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

CANDIDATES = [
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1-mini",
    "openai/o3-mini",
    "openai/o3",
]

for m in CANDIDATES:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with one word."}],
            max_tokens=5,
        )
        u = resp.usage
        print(f"  ✅ {m:<25}  prompt={u.prompt_tokens} completion={u.completion_tokens}")
    except Exception as e:
        msg = str(e)[:120]
        print(f"  ❌ {m:<25}  {type(e).__name__}: {msg}")
