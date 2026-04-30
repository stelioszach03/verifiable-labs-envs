"""HTTP route handlers.

Each module exposes a ``router`` (FastAPI APIRouter). The top-level
``main.py`` mounts them at ``/`` (health) and ``/v1/`` (everything else).
"""
