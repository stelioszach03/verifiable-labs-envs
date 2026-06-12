"""scripts/publish/bump_versions.py — bump version in every in-repo
``pyproject.toml``.

PyPI rejects a re-upload of an already-published version (and yanking
preserves history without freeing the slot), so a fresh upload needs
a new version. This script writes a new version into every
in-repo package's ``pyproject.toml`` so a single ``publish.sh --all``
pass goes through.

Default behaviour: bump the *patch* component, preserving any
pre-release suffix (e.g. ``0.1.0a1`` → ``0.1.0a2``, ``0.1.0`` →
``0.1.1``, ``1.0.0`` → ``1.0.1``).

Override with ``--set X.Y.Z`` to set the same explicit version on
every package, or ``--package NAME --set X.Y.Z`` for a single one.

Usage:
    python scripts/publish/bump_versions.py --dry-run        # show diff
    python scripts/publish/bump_versions.py --apply          # write
    python scripts/publish/bump_versions.py --set 0.1.0a2 --apply
    python scripts/publish/bump_versions.py --package verifiable-labs-envs \
            --set 0.1.0a2 --apply
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_pypi_helpers", SCRIPT_DIR / "_pypi_helpers.py"
)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)


VERSION_RE = re.compile(
    r'(^version\s*=\s*[\'"])([^\'"]+)([\'"])', re.MULTILINE
)


def _substitute_version(text: str, new_version: str) -> tuple[str, int]:
    """Replace the first ``version = "..."`` line in ``text`` with the
    requested value, preserving the original quote style. Returns
    ``(new_text, count)`` like :meth:`re.subn`.

    Lifted out of the bump loop so ruff doesn't flag a closure-over-
    loop-variable, and so the replacement is a plain function instead
    of a lambda.
    """
    def _replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{new_version}{match.group(3)}"

    return VERSION_RE.subn(_replace, text, count=1)


def _bump_patch(v: str) -> str:
    """Bump the patch component, preserving any pre-release suffix.

    Examples:
      0.1.0      → 0.1.1
      0.1.0a1    → 0.1.0a2
      0.1.0a4    → 0.1.0a5
      1.0.0      → 1.0.1
      0.3.0      → 0.3.1
    """
    # Pre-release suffix? (a/b/rc/.devN)
    for prefix in ("a", "b", "rc", ".dev"):
        if prefix in v:
            head, _, tail = v.partition(prefix)
            try:
                tail_n = int(tail)
            except ValueError:
                # Unknown suffix shape; fall through to numeric bump.
                break
            return f"{head}{prefix}{tail_n + 1}"

    parts = v.split(".")
    if len(parts) < 3 or not all(p.isdigit() for p in parts[:3]):
        raise ValueError(f"don't know how to bump version {v!r}")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the new version to disk (default: dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change but do not write.",
    )
    parser.add_argument(
        "--package",
        default=None,
        help="Limit to one package (default: every in-repo package).",
    )
    parser.add_argument(
        "--set",
        dest="explicit",
        default=None,
        help="Set every selected package to this exact version "
        "(skips the patch-bump heuristic).",
    )
    ns = parser.parse_args(argv)

    pkgs = helpers.discover_packages()
    if ns.package:
        pkgs = [p for p in pkgs if p.name == ns.package]
        if not pkgs:
            helpers.err(f"ERROR: --package {ns.package} not found in repo.")
            return 2

    apply = ns.apply and not ns.dry_run

    print(f"# bumping {len(pkgs)} package(s)  apply={apply}")
    n_changed = 0
    for p in pkgs:
        new_version = ns.explicit or _bump_patch(p.version)
        if new_version == p.version:
            print(f"  · {p.name}  {p.version} (no change)")
            continue
        print(f"  · {p.name}  {p.version}  →  {new_version}")
        if apply:
            text = p.pyproject_path.read_text(encoding="utf-8")
            new_text, n = _substitute_version(text, new_version)
            if n != 1:
                helpers.err(
                    f"    ⚠ couldn't substitute version in "
                    f"{p.pyproject_path}; skipped"
                )
                continue
            p.pyproject_path.write_text(new_text, encoding="utf-8")
        n_changed += 1
    print(f"# summary  changed={n_changed}  apply={apply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
