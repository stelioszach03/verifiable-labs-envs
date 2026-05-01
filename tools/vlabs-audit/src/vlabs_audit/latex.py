"""LaTeX/PDF report rendering — Jinja2 templates + ``tectonic`` compile.

Two layers, one for testability:

* :func:`render_tex` — pure string output. Builds a context dict from
  an :class:`vlabs_audit.stats.AuditStats`, renders each section
  template to a string, and inlines those strings into ``main.tex.j2``.
  No filesystem writes outside the caller-controlled scope, no
  subprocess invocations.
* :func:`render_report` — the full pipeline. Calls :func:`render_tex`,
  copies the rendered ``main.tex`` plus ``vlabs_report.cls`` and the
  caller's figure PDFs into a temporary working directory, runs
  ``tectonic -X compile main.tex``, then moves the produced ``main.pdf``
  to the requested output path.

LaTeX escaping
--------------
The Jinja environment registers ``e`` / ``latex`` filters that escape
the eleven LaTeX special characters listed in the Phase 17 plan
(``\\``, ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``, ``^``, ``~``,
``<``, ``>``). Always run user-supplied strings through this filter
before splicing them into a template.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    StrictUndefined,
    select_autoescape,
)

from vlabs_audit import __version__ as VLABS_AUDIT_VERSION
from vlabs_audit.stats import AuditStats

# ── template root resolution ─────────────────────────────────────────
# Editable installs see the source tree under
# ``tools/vlabs-audit/templates/``; wheel installs see them inside the
# package at ``vlabs_audit/templates/`` via the ``force-include`` rule
# in ``pyproject.toml``.


def _find_templates_dir() -> Path:
    src_tree = Path(__file__).resolve().parent.parent.parent / "templates"
    if src_tree.exists():
        return src_tree
    in_package = Path(__file__).resolve().parent / "templates"
    if in_package.exists():
        return in_package
    raise RuntimeError(
        "vlabs-audit: LaTeX templates not found. Searched "
        f"{src_tree} and {in_package}. Reinstall `vlabs-audit`."
    )


_TEMPLATES_DIR = _find_templates_dir()
_STYLE_DIR = _TEMPLATES_DIR / "style"
_CLASS_FILE = _STYLE_DIR / "vlabs_report.cls"

_TECTONIC_INSTALL_URL = (
    "https://tectonic-typesetting.github.io/en-US/install.html"
)

# ── per-env metadata ─────────────────────────────────────────────────

_ENV_DISPLAY: dict[str, str] = {
    "sparse-fourier-recovery": "Sparse Fourier Recovery",
    "sparse-fourier-recovery-multiturn": "Sparse Fourier Recovery (multi-turn)",
    "sparse-fourier-recovery-tools": "Sparse Fourier Recovery (with tools)",
    "phase-retrieval": "Phase Retrieval",
    "phase-retrieval-multiturn": "Phase Retrieval (multi-turn)",
    "super-resolution-div2k-x4": "Image Super-Resolution (DIV2K, ×4)",
    "lodopab-ct-simplified": "Low-Dose CT Reconstruction",
    "lodopab-ct-simplified-multiturn": "Low-Dose CT Reconstruction (multi-turn)",
    "mri-knee-reconstruction": "Accelerated MRI Reconstruction (knee)",
    "mri-knee-reconstruction-multiturn": "Accelerated MRI Reconstruction (knee, multi-turn)",
}

_ENV_DESCRIPTION: dict[str, str] = {
    "sparse-fourier-recovery": (
        "Recover a k-sparse complex signal from a small number of "
        "compressive Fourier measurements. The model returns the support "
        "indices and complex amplitudes; the environment scores normalised "
        "mean-squared error against the ground-truth signal and reports a "
        "conformal interval over the reconstruction quality."
    ),
    "phase-retrieval": (
        "Recover the phase of a complex signal from magnitude-only "
        "measurements (a classic ill-posed inverse problem in optics and "
        "imaging). The environment scores spectral overlap with the "
        "ground-truth signal under the standard global-phase ambiguity."
    ),
    "super-resolution-div2k-x4": (
        "4× single-image super-resolution on the DIV2K validation set. "
        "The environment compares structural similarity between the "
        "reconstructed and ground-truth high-resolution images."
    ),
    "lodopab-ct-simplified": (
        "Low-dose CT image reconstruction from sinogram measurements "
        "(based on the LoDoPaB-CT benchmark)."
    ),
    "mri-knee-reconstruction": (
        "Accelerated MRI reconstruction from sub-Nyquist k-space "
        "measurements on the fastMRI knee dataset."
    ),
}


def _env_display(env: str) -> str:
    return _ENV_DISPLAY.get(env, env.replace("-", " ").title())


def _env_description(env: str) -> str:
    return _ENV_DESCRIPTION.get(
        env,
        "A Verifiable Labs scientific-reasoning environment with a "
        "conformal-calibrated reward.",
    )


# ── LaTeX escaping ───────────────────────────────────────────────────

_LATEX_BACKSLASH_SENTINEL = "LATEXBSLASH"

_LATEX_PAIRS_AFTER_BACKSLASH: tuple[tuple[str, str], ...] = (
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("^", r"\^{}"),
    ("~", r"\~{}"),
    ("<", r"\textless{}"),
    (">", r"\textgreater{}"),
)


def latex_escape(value: object) -> str:
    """Escape a value for safe inclusion in a LaTeX document.

    Handles the eleven special characters listed in the Phase 17 plan:
    ``\\``, ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``, ``^``,
    ``~``, ``<``, ``>``. ``None`` becomes the empty string; everything
    else is rendered with :func:`str` first.
    """
    if value is None:
        return ""
    s = str(value)
    # Backslash first via sentinel so subsequent replacements can't
    # interfere with the multi-char ``\textbackslash{}`` substitution.
    s = s.replace("\\", _LATEX_BACKSLASH_SENTINEL)
    for src, dst in _LATEX_PAIRS_AFTER_BACKSLASH:
        s = s.replace(src, dst)
    return s.replace(_LATEX_BACKSLASH_SENTINEL, r"\textbackslash{}")


# ── Jinja env ────────────────────────────────────────────────────────


def _make_env(loader: BaseLoader | None = None) -> Environment:
    env = Environment(
        loader=loader or FileSystemLoader(_TEMPLATES_DIR),
        autoescape=select_autoescape(disabled_extensions=("tex", "j2")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["latex"] = latex_escape
    env.filters["e"] = latex_escape
    return env


# ── headline + recommendations ───────────────────────────────────────


def _headline_finding(stats: AuditStats) -> str:
    """One short paragraph summarising the audit's headline numbers."""
    target = 1.0 - stats.alpha
    cov = stats.aggregate_coverage_holdout
    cov_clause = (
        f"Empirical (held-out) coverage averaged "
        f"{cov * 100:.1f}\\% across all environments versus a target of "
        f"{target * 100:.0f}\\%."
        if cov is not None
        else "Empirical coverage data was not available for every environment."
    )
    parse_pct = stats.aggregate_parse_failure_rate * 100
    return (
        f"Across the {len(stats.per_env)} environment"
        f"{'' if len(stats.per_env) == 1 else 's'} audited, "
        f"the model achieved a mean reward of "
        f"\\textbf{{{stats.aggregate_mean_reward:.3f}}} "
        f"with a 95\\% bootstrap CI of "
        f"$[{stats.aggregate_ci_low:.3f},\\, {stats.aggregate_ci_high:.3f}]$ "
        f"and a parse-failure rate of {parse_pct:.1f}\\%. "
        f"{cov_clause}"
    )


def _recommendations(stats: AuditStats) -> list[dict[str, str]]:
    """Auto-generate a short list of conservative recommendations."""
    target = 1.0 - stats.alpha
    out: list[dict[str, str]] = []
    for es in stats.per_env:
        env_t = latex_escape(es.env)
        if es.mean_reward < 0.4:
            out.append({
                "text_tex": (
                    f"Mean reward on \\texttt{{{env_t}}} "
                    f"is {es.mean_reward:.3f}, at or below the 0.4 threshold "
                    f"typical for non-trivial models on this task class. "
                    f"A larger or chain-of-thought-prompted model may be "
                    f"required before deployment in this environment."
                ),
            })
        if es.coverage_holdout is not None and es.coverage_holdout < target:
            out.append({
                "text_tex": (
                    f"Empirical coverage on \\texttt{{{env_t}}} "
                    f"({es.coverage_holdout:.3f}) falls below the target "
                    f"({target:.2f}). The most likely cause is calibration "
                    f"sample-size insufficiency; rerun with "
                    f"\\texttt{{--episodes}} doubled before drawing "
                    f"calibration conclusions."
                ),
            })
        if es.parse_failure_rate > 0.05:
            out.append({
                "text_tex": (
                    f"Parse-failure rate on \\texttt{{{env_t}}} "
                    f"({es.parse_failure_rate * 100:.1f}\\%) exceeds 5\\%. "
                    f"Tightening the format-compliance instructions in the "
                    f"prompt will recover most of these episodes without "
                    f"changing the underlying model."
                ),
            })
    if not out:
        out.append({
            "text_tex": (
                "Every environment passes both coverage calibration and "
                "parse-quality thresholds. The model performs in line with "
                "expectations for this audit regime; no specific remedial "
                "action is suggested."
            ),
        })
    return out


def _recommendation_summary(stats: AuditStats) -> str:
    recs = _recommendations(stats)
    if len(recs) == 1 and "no specific remedial" in recs[0]["text_tex"]:
        return (
            "No remedial action is recommended at this audit's sample "
            "size and $\\alpha$ level — see Section~5 for the detailed list."
        )
    return (
        f"{len(recs)} actionable recommendation"
        f"{'' if len(recs) == 1 else 's'} are listed in Section~5; "
        f"the most consequential are summarised in the per-environment "
        f"results that follow."
    )


# ── status helpers ───────────────────────────────────────────────────


def _coverage_status_tex(coverage: float | None, target: float) -> str:
    """Render the per-env coverage status cell as escaped LaTeX."""
    if coverage is None:
        return r"\textcolor{vlabsmidgrey}{n/a}"
    if coverage >= target:
        return r"\textcolor{vlabsdarkblue}{\textbf{within target}}"
    return r"\textbf{below target}"


def _coverage_str(coverage: float | None) -> str:
    return "n/a" if coverage is None else f"{coverage:.3f}"


# ── context build ────────────────────────────────────────────────────


def _build_context(stats: AuditStats, *, now: _dt.datetime | None = None) -> dict[str, Any]:
    """Construct the dict the templates consume.

    All values are pre-escaped where they will land inside running text
    (suffix ``_tex``); fields ending in ``_str`` are already-formatted
    numerics (safe to splice as-is); fields ending in ``_raw`` are
    intentionally non-escaped (e.g. JSON dumps that go inside
    ``verbatim``).
    """
    now = now or _dt.datetime.now(_dt.UTC)
    target = 1.0 - stats.alpha

    envs_ctx: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for es in stats.per_env:
        envs_ctx.append({
            "env": es.env,
            "env_tex": latex_escape(es.env),
            "env_display": _env_display(es.env),
            "env_display_tex": latex_escape(_env_display(es.env)),
            "description_tex": latex_escape(_env_description(es.env)),
            "n_episodes": es.n_episodes,
            "n_success": es.n_success,
            "n_failed": es.n_failed,
            "mean_reward_str": f"{es.mean_reward:.3f}",
            "ci_str": f"$[{es.ci_low:.3f},\\, {es.ci_high:.3f}]$",
            "parse_failure_pct_str": f"{es.parse_failure_rate * 100:.1f}\\%",
            "format_valid_pct_str": f"{es.format_valid_rate * 100:.1f}\\%",
            "coverage_holdout_str": _coverage_str(es.coverage_holdout),
            "figure_filename": f"reward_distribution_{_safe_env_slug(es.env)}.pdf",
            "figure_filename_tex": latex_escape(
                f"reward_distribution_{_safe_env_slug(es.env)}.pdf"
            ),
        })
        coverage_rows.append({
            "env_tex": latex_escape(_env_display(es.env)),
            "coverage_str": _coverage_str(es.coverage_holdout),
            "target_str": f"{target:.3f}",
            "status_tex": _coverage_status_tex(es.coverage_holdout, target),
        })

    total_episodes = sum(es.n_episodes for es in stats.per_env)
    audit_id_short = stats.audit_id.removeprefix("aud_")[:8]

    stats_json = json.dumps(stats.model_dump(), indent=2, default=str)

    return {
        # Identity / metadata
        "title": "Verifiable Labs",
        "title_tex": latex_escape("Verifiable Labs"),
        "model": stats.model,
        "model_tex": latex_escape(stats.model),
        "model_display": stats.model,
        "model_display_tex": latex_escape(stats.model),
        "model_cite": stats.model.replace("_", "-"),
        "audit_id": stats.audit_id,
        "audit_id_tex": latex_escape(stats.audit_id),
        "audit_id_short": audit_id_short,
        "date": now.date().isoformat(),
        "date_tex": latex_escape(now.date().isoformat()),
        "year": now.year,
        "timestamp_tex": latex_escape(now.isoformat(timespec="seconds")),
        # Settings
        "alpha": stats.alpha,
        "alpha_str": f"{stats.alpha:.2f}",
        "target_coverage": target,
        "target_coverage_str": f"{target:.2f}",
        "target_coverage_pct": f"{target * 100:.0f}",
        "n_envs": len(stats.per_env),
        "n_episodes_per_env": stats.n_episodes_per_env,
        "total_episodes": total_episodes,
        "seed_start": _seed_start_or_zero(stats),
        # Aggregate text
        "headline_finding_tex": _headline_finding(stats),
        "recommendation_summary_tex": _recommendation_summary(stats),
        "recommendations": _recommendations(stats),
        # Per-env
        "envs": envs_ctx,
        "coverage_rows": coverage_rows,
        # Reproducibility / tooling
        "vlabs_audit_version_tex": latex_escape(VLABS_AUDIT_VERSION),
        "vlabs_calibrate_version_tex": latex_escape(_vlabs_calibrate_version()),
        "python_version_tex": latex_escape(_python_version_short()),
        "stats_json_raw": stats_json,
    }


def _safe_env_slug(env: str) -> str:
    out = []
    for ch in env:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _python_version_short() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro} ({platform.python_implementation()})"


def _vlabs_calibrate_version() -> str:
    try:
        from vlabs_calibrate import __version__ as v  # type: ignore[import]
        return str(v)
    except Exception:  # noqa: BLE001 — best-effort, never fail the render
        return "unknown"


def _seed_start_or_zero(stats: AuditStats) -> int:
    """The audit row carries the original config; pull seed_start out of it."""
    # The renderer doesn't have direct access to the audit row, but the
    # caller can pass us a populated EnvStats; for now we surface 0 as a
    # safe default. CLI overrides via a custom context if needed.
    _ = stats  # currently unused; placeholder for future per-call seed forwarding.
    return 0


# ── render: text-only ────────────────────────────────────────────────


def render_tex(stats: AuditStats, figures_dir: Path | None = None) -> str:
    """Render the audit's full LaTeX source as a single string.

    ``figures_dir`` is accepted for API symmetry but is not consulted —
    the templates reference figures by basename so they line up with
    files copied into tectonic's working directory by :func:`render_report`.
    """
    del figures_dir  # noqa: F841 — kept for API symmetry, see docstring
    env = _make_env()
    context = _build_context(stats)

    sections = {}
    for name in ("exec_summary", "methodology", "results", "recommendations", "appendix"):
        tpl = env.get_template(f"sections/{name}.tex.j2")
        sections[f"{name}_tex"] = tpl.render(**context)

    main_tpl = env.get_template("main.tex.j2")
    return main_tpl.render(**context, **sections)


# ── render: full pipeline (tectonic) ────────────────────────────────


def render_report(
    stats: AuditStats,
    figures_dir: Path,
    output_path: Path,
    *,
    timeout_s: float = 180.0,
) -> Path:
    """Render the audit to a PDF at ``output_path`` via ``tectonic``.

    Steps: render LaTeX source → write to a tempdir along with the class
    file and every PDF in ``figures_dir`` → run ``tectonic -X compile`` →
    move ``main.pdf`` to ``output_path``. Raises :class:`RuntimeError`
    when ``tectonic`` is missing (with the install URL) or when the
    LaTeX compile fails (with the tail of the tectonic log).
    """
    figures_dir = Path(figures_dir)
    output_path = Path(output_path)

    if shutil.which("tectonic") is None:
        raise RuntimeError(
            "vlabs-audit: `tectonic` is not on PATH. Install it from "
            f"{_TECTONIC_INSTALL_URL} (e.g. `cargo install tectonic` or "
            f"`brew install tectonic`)."
        )

    tex_source = render_tex(stats)

    with tempfile.TemporaryDirectory(prefix="vlabs-audit-tex-") as work_dir_s:
        work_dir = Path(work_dir_s)
        (work_dir / "main.tex").write_text(tex_source, encoding="utf-8")
        # Class file alongside main.tex.
        if _CLASS_FILE.exists():
            shutil.copy(_CLASS_FILE, work_dir / _CLASS_FILE.name)
        else:
            raise RuntimeError(
                f"vlabs-audit: bundled LaTeX class missing at {_CLASS_FILE}; "
                f"reinstall `vlabs-audit`."
            )
        # Figures: copy every *.pdf so the relative \includegraphics paths resolve.
        if figures_dir.exists():
            for pdf in figures_dir.glob("*.pdf"):
                shutil.copy(pdf, work_dir / pdf.name)

        proc = subprocess.run(  # noqa: S603 — argv fully controlled
            ["tectonic", "-X", "compile", "main.tex"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"vlabs-audit: tectonic compile failed (exit {proc.returncode}). "
                f"Last 2000 chars of log:\n{tail}"
            )

        pdf_src = work_dir / "main.pdf"
        if not pdf_src.exists():
            raise RuntimeError(
                "vlabs-audit: tectonic exited 0 but produced no main.pdf."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf_src), str(output_path))

    return output_path


__all__: Iterable[str] = [
    "latex_escape",
    "render_report",
    "render_tex",
]
