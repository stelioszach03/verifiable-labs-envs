#!/usr/bin/env python3
"""scripts/deploy/generate_fernet_key.py — one-shot Fernet keygen.

Prints a fresh 44-char base64 Fernet key on stdout. Stelios runs once
during provisioning, then pastes the output at the
``VLABS_DATA_LLM_KEY_ENCRYPTION`` prompt of
``scripts/deploy/provision_production_secrets.sh``.

The output is ALL the script writes — no logging, no extra commentary
— so it can be piped straight into the clipboard:

    python scripts/deploy/generate_fernet_key.py | pbcopy   # macOS
    python scripts/deploy/generate_fernet_key.py | xclip -selection clipboard  # Linux

The produced key is single-use:
- Encrypts ``dataset_jobs.llm_api_key_encrypted`` (Phase 23 R11
  mitigation: customer LLM keys are stored encrypted at rest).
- Once the production key is set, it must NEVER be rotated
  in-place — rotation would invalidate every existing encrypted
  payload. A future rotation procedure (deferred to Phase 31.G)
  will add a key-id versioning column.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        print(
            "ERROR: cryptography package not installed. "
            "Run: pip install cryptography",
            file=sys.stderr,
        )
        return 1

    key = Fernet.generate_key().decode("ascii")
    # Single-line output → pipe into clipboard or paste at prompt.
    sys.stdout.write(key + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
