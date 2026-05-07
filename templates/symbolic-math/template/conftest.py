"""pytest configuration for the __ENV_ID__ env scaffold.

Adds this directory to ``sys.path`` so the tests can import the
``__ENV_PY__`` package by its bare module name without requiring an
editable install. Lets ``pytest tests/`` work straight after
``scripts/create_env.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
