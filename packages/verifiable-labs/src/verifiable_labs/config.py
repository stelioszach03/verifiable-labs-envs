"""API-key handling for the CLI.

Two sources, in priority order:

1. Process environment variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
   ``GOOGLE_API_KEY``).
2. ``~/.verifiable/config.toml`` written by ``verifiable login``.

The TOML file is created with mode 0600 so only the current user can read it.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".verifiable"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("google", "GOOGLE_API_KEY"),
)


@dataclass
class Config:
    anthropic: str = ""
    openai: str = ""
    google: str = ""


def _read_toml() -> dict[str, str]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}
    raw = data.get("api_keys", {})
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def load_config() -> Config:
    """Resolve all known keys. Env vars beat the TOML file."""
    file_keys = _read_toml()
    cfg = Config()
    cfg.anthropic = os.environ.get("ANTHROPIC_API_KEY", "") or file_keys.get("anthropic", "")
    cfg.openai = os.environ.get("OPENAI_API_KEY", "") or file_keys.get("openai", "")
    cfg.google = os.environ.get("GOOGLE_API_KEY", "") or file_keys.get("google", "")
    return cfg


def require_key(provider: str) -> str:
    """Return the key for ``provider`` or raise SystemExit with instructions."""
    cfg = load_config()
    key = getattr(cfg, provider, "")
    env_var = next((var for prov, var in _PROVIDERS if prov == provider), provider.upper() + "_API_KEY")
    if key:
        return key
    raise SystemExit(
        f"\n❌ {env_var} not set\n"
        f"\nFix:\n"
        f"  export {env_var}=your_key_here\n"
        f"\nOr run:\n"
        f"  verifiable login\n"
    )


def interactive_setup() -> int:
    """Run the prompt-driven setup. Writes ``~/.verifiable/config.toml`` 0600."""
    print("Verifiable Labs · API key setup")
    print("=" * 40)
    print()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_toml()
    keys: dict[str, str] = {}

    for provider, env_var in _PROVIDERS:
        env_value = os.environ.get(env_var, "").strip()
        prior = existing.get(provider, "")
        current = env_value or prior

        if current:
            print(f"{provider.title()} ({env_var}): set, ending in …{current[-4:]}")
            answer = input("  Update? [y/N]: ").strip().lower()
            if answer != "y":
                if prior:
                    keys[provider] = prior
                continue

        try:
            new_key = input(f"{provider.title()} API key (Enter to skip): ").strip()
        except EOFError:
            new_key = ""
        if new_key:
            keys[provider] = new_key

    # Write atomically with 0600 perms.
    tmp = CONFIG_FILE.with_suffix(".toml.tmp")
    with open(tmp, "w") as f:
        f.write("[api_keys]\n")
        for provider, _env in _PROVIDERS:
            if keys.get(provider):
                f.write(f'{provider} = "{keys[provider]}"\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG_FILE)

    print()
    print(f"✓ Saved to {CONFIG_FILE} (mode 0600)")

    if not keys:
        print("  (no keys provided — set env vars or rerun `verifiable login`)", file=sys.stderr)
    return 0
