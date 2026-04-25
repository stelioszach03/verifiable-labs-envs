#!/usr/bin/env python3
"""Scaffold a new Verifiable Labs inverse-problem RL environment.

Reads ``templates/inverse-problem/template/`` and writes a rendered
copy under ``environments/<env_py>/`` (or wherever ``--target``
points). Replaces literal placeholders (``__ENV_ID__``, ``__ENV_PY__``,
``__ENV_CLASS__``, ``__DOMAIN__``, ``__DOMAIN_TAG__``) in every
template file and renames the ``__ENV_PY__/`` template directory to
the new env's Python module name.

Usage::

    python scripts/create_env.py seismic-fwi --domain "geophysics"
    python scripts/create_env.py seismic-fwi --domain "geophysics" \\
        --target /tmp/scratch-env-tree

After scaffolding, edit the four ``NotImplementedError`` stubs and
run ``python scripts/validate_env.py environments/<env_py>``.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "templates" / "inverse-problem" / "template"
TEMPLATE_PY_MARKER = "__ENV_PY__"

_KEBAB = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
# File suffixes whose contents we substitute through. Anything not in
# this list is copied byte-for-byte.
_TEXT_SUFFIXES = {
    ".py", ".pyi", ".toml", ".md", ".txt", ".cfg", ".ini",
    ".yml", ".yaml", ".json", ".rst",
}


def _kebab_to_snake(name: str) -> str:
    return name.replace("-", "_")


def _kebab_to_camel(name: str) -> str:
    return "".join(part.title() for part in name.split("-"))


def _slugify(text: str) -> str:
    """Slugify a free-form domain name to a kebab-case tag."""
    text = text.strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^a-z0-9-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "general"


def _build_substitutions(env_id: str, domain: str) -> dict[str, str]:
    env_py = _kebab_to_snake(env_id)
    return {
        "__ENV_ID__": env_id,
        "__ENV_PY__": env_py,
        "__ENV_CLASS__": _kebab_to_camel(env_id) + "Env",
        "__DOMAIN__": domain,
        "__DOMAIN_TAG__": _slugify(domain),
    }


def _apply_substitutions(text: str, subs: dict[str, str]) -> str:
    for marker, value in subs.items():
        text = text.replace(marker, value)
    return text


def _is_text_file(path: Path) -> bool:
    return path.suffix in _TEXT_SUFFIXES


def _render_tree(
    src_dir: Path,
    dst_dir: Path,
    subs: dict[str, str],
) -> list[Path]:
    """Recursively copy ``src_dir`` to ``dst_dir`` with substitutions.

    Renames any path component matching a substitution marker.
    Returns the list of files written.
    """
    written: list[Path] = []
    for src in sorted(src_dir.rglob("*")):
        rel = src.relative_to(src_dir)
        # Apply substitutions to each path component.
        rendered_parts = [_apply_substitutions(part, subs) for part in rel.parts]
        dst = dst_dir.joinpath(*rendered_parts)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _is_text_file(src):
            content = src.read_text(encoding="utf-8")
            dst.write_text(_apply_substitutions(content, subs), encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        written.append(dst)
    return written


def _validate_env_id(env_id: str) -> None:
    if not _KEBAB.match(env_id):
        raise SystemExit(
            f"Invalid env id {env_id!r}: must be kebab-case "
            "(lowercase letters / digits / hyphens, must start with a letter, "
            "no consecutive hyphens). Examples: 'seismic-fwi-1d', "
            "'phase-retrieval', 'protein-distogram'."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("env_id", help="kebab-case env slug, e.g. seismic-fwi")
    parser.add_argument(
        "--domain", required=True,
        help="human-readable domain, e.g. 'geophysics' or 'medical-imaging'",
    )
    parser.add_argument(
        "--target", default=None,
        help="target directory (default: <repo>/environments/<env_py>)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing target directory",
    )
    args = parser.parse_args()

    _validate_env_id(args.env_id)
    if not TEMPLATE_DIR.is_dir():
        raise SystemExit(f"template directory not found: {TEMPLATE_DIR}")

    subs = _build_substitutions(args.env_id, args.domain)
    env_py = subs["__ENV_PY__"]

    if args.target is None:
        target = REPO_ROOT / "environments" / env_py
    else:
        target = Path(args.target).resolve()

    if target.exists():
        if not args.force:
            raise SystemExit(
                f"target {target} already exists. Pass --force to overwrite."
            )
        shutil.rmtree(target)
    target.mkdir(parents=True)

    written = _render_tree(TEMPLATE_DIR, target, subs)
    rel = target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target

    print(f"scaffolded {len(written)} files into {rel}/")
    print()
    print("next steps:")
    print(f"  1. fill the NotImplementedError stubs in {rel}/{env_py}/")
    print(f"  2. python scripts/validate_env.py {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
