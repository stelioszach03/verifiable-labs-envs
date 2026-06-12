"""scripts/publish/_pypi_helpers.py — shared helpers for the publish
toolkit.

Pure-python; depends only on httpx (already in the api dev set). No
secret values are returned or printed; all token reads happen via
``os.environ.get`` and are passed straight to twine / requests.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Packages that are intentionally NEVER published to PyPI.
EXCLUDED_PACKAGES: frozenset[str] = frozenset({
    # vlabs-api is the deployed FastAPI service, not a redistributable
    # library. Publishing it would expose internal layout + ship test
    # fixtures. Stays private; deploys via Docker.
    "vlabs-api",
})

PYPI_INDEX_URL = "https://pypi.org/pypi"
TEST_PYPI_INDEX_URL = "https://test.pypi.org/pypi"


@dataclass(frozen=True)
class PackageInfo:
    name: str  # canonical PyPI name (e.g. "verifiable-labs-envs")
    version: str  # local version from pyproject.toml
    pyproject_path: Path
    package_dir: Path  # parent dir of pyproject.toml


def discover_packages() -> list[PackageInfo]:
    """Walk the repo for every ``pyproject.toml`` and return canonical
    package info. Skips :data:`EXCLUDED_PACKAGES`.

    Locations probed (in order):
      - root pyproject.toml
      - packages/*/pyproject.toml
      - tools/*/pyproject.toml
      - services/*/pyproject.toml (excluded by default)
    """
    candidates: list[Path] = [
        REPO_ROOT / "pyproject.toml",
        *sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")),
        *sorted((REPO_ROOT / "tools").glob("*/pyproject.toml")),
        *sorted((REPO_ROOT / "services").glob("*/pyproject.toml")),
    ]

    out: list[PackageInfo] = []
    name_re = re.compile(r'^name\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE)
    version_re = re.compile(
        r'^version\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE
    )

    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        nm = name_re.search(text)
        ver = version_re.search(text)
        if not nm or not ver:
            continue
        name = nm.group(1)
        version = ver.group(1)
        if name in EXCLUDED_PACKAGES:
            continue
        out.append(
            PackageInfo(
                name=name,
                version=version,
                pyproject_path=path,
                package_dir=path.parent,
            )
        )
    return out


def query_pypi_versions(
    name: str, *, index: str = PYPI_INDEX_URL, timeout: float = 10.0
) -> list[str]:
    """Return all published versions of ``name`` on the chosen index.

    Empty list when the package isn't on the index yet. Raises
    :class:`RuntimeError` on network / parse errors.
    """
    import httpx

    url = f"{index}/{name}/json"
    try:
        r = httpx.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"network: {type(exc).__name__}: {exc}") from exc
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code} from {url}")
    data = r.json()
    return sorted(data.get("releases", {}).keys())


def query_pypi_yanked_state(
    name: str, *, index: str = PYPI_INDEX_URL, timeout: float = 10.0
) -> dict[str, bool]:
    """Map ``version -> is_yanked`` for the package on the index.

    Empty dict when the package isn't on the index. Raises
    :class:`RuntimeError` on network / parse errors.
    """
    import httpx

    url = f"{index}/{name}/json"
    try:
        r = httpx.get(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"network: {type(exc).__name__}: {exc}") from exc
    if r.status_code == 404:
        return {}
    if r.status_code != 200:
        raise RuntimeError(f"http {r.status_code} from {url}")
    data = r.json()
    out: dict[str, bool] = {}
    for ver, files in (data.get("releases") or {}).items():
        # PyPI marks every artifact for a yanked version with
        # ``"yanked": true``. Use any() so a partial yank is treated as
        # "yanked from default resolution".
        out[ver] = any(bool(f.get("yanked", False)) for f in (files or []))
    return out


def get_token(*, prod: bool) -> str | None:
    """Return the token for the chosen index from the env. Returns
    None when the user opted to skip that index in the loader."""
    var = "PYPI_API_TOKEN" if prod else "TEST_PYPI_API_TOKEN"
    val = os.environ.get(var, "").strip()
    return val or None


def index_url(*, prod: bool) -> str:
    return PYPI_INDEX_URL if prod else TEST_PYPI_INDEX_URL


def upload_repository(*, prod: bool) -> str:
    """The twine ``--repository-url`` value for the chosen index."""
    return (
        "https://upload.pypi.org/legacy/"
        if prod
        else "https://test.pypi.org/legacy/"
    )


def err(msg: str) -> None:
    """Write a single-line message to stderr."""
    print(msg, file=sys.stderr)
