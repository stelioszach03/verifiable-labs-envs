"""Entry point for ``python -m verifiable_labs_envs``.

Mirrors the ``verifiable`` console script so the CLI works in
environments where the script entry point hasn't been wired up.
"""
from verifiable_labs_envs.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
