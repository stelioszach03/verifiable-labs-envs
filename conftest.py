"""Pytest collection-time bootstrap: ensure src/ is importable.

Hatchling's editable install occasionally writes the ``.pth`` path entry without
a terminating newline, which Python's site module silently ignores. This keeps
pytest runs reproducible without depending on the broken ``.pth`` file.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
