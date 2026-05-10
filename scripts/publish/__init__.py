"""PyPI publishing toolkit.

Mirrors the silent-prompt + secrets-stay-local convention used by
``scripts/deploy/provision_production_secrets.sh``: no PyPI tokens
ever appear in chat, logs, or committed files. Use:

    source scripts/publish/_load_pypi_secrets.sh
    bash   scripts/publish/publish.sh --list

then ``--test`` (default) for test.pypi.org or ``--prod`` for the
real PyPI.
"""
