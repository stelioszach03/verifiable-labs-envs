"""``vlabs-audit`` Typer CLI entry point.

Sub-stage 17.A surface area: ``audit`` (with ``--dry-run``) and ``version``.
17.B wires the ``audit`` command to :class:`vlabs_audit.runner.EpisodeRunner`,
which schedules + drains episodes against the local SQLite store. 17.C+D
add stats + figures behind ``--print-stats`` / ``--figures-dir``. 17.E
renders a LaTeX/PDF report via ``tectonic`` whenever ``--output`` points
to a real path.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from vlabs_audit import __version__
from vlabs_audit.anonymize import (
    anonymize_audit_stats,
    parse_anonymize_labels,
    resolve_anonymize_label,
)
from vlabs_audit.config import AuditConfig, load_config
from vlabs_audit.figures import render_all_figures
from vlabs_audit.latex import render_report
from vlabs_audit.runner import EpisodeRunner
from vlabs_audit.stats import compute_audit_stats, format_stats_table
from vlabs_audit.storage import AuditStore

app = typer.Typer(
    name="vlabs-audit",
    help="Generate capability reports for any frontier model on Verifiable Labs envs.",
    no_args_is_help=True,
    add_completion=False,
)


def _parse_envs(value: str | None) -> list[str] | None:
    """Split a comma-separated env list, trimming whitespace."""
    if value is None:
        return None
    return [e.strip() for e in value.split(",") if e.strip()]


def _format_config(cfg: AuditConfig) -> str:
    total = cfg.episodes * len(cfg.envs)
    lines = [
        f"vlabs-audit v{__version__} — dry run",
        "",
        f"  model:           {cfg.model}",
        f"  envs:            {', '.join(cfg.envs)}",
        f"  episodes / env:  {cfg.episodes}",
        f"  total episodes:  {total}",
        f"  alpha:           {cfg.alpha}  (target coverage {1 - cfg.alpha:.2f})",
        f"  output:          {cfg.output}",
        f"  parallel:        {cfg.parallel}",
        f"  seed_start:      {cfg.seed_start}",
        f"  anonymize:       {cfg.anonymize} (label: {cfg.anonymize_label!r})",
        "",
        "(dry run — no episodes executed, no API calls made)",
    ]
    return "\n".join(lines)


@app.command()
def audit(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="YAML config file path."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Model id (overrides config)."),
    ] = None,
    envs: Annotated[
        str | None,
        typer.Option(help="Comma-separated env ids (overrides config)."),
    ] = None,
    episodes: Annotated[
        int | None,
        typer.Option(help="Episodes per env (overrides config)."),
    ] = None,
    alpha: Annotated[
        float | None,
        typer.Option(help="Conformal alpha in (0, 1) (overrides config)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="PDF output path (overrides config)."),
    ] = None,
    parallel: Annotated[
        int | None,
        typer.Option(help="Parallel workers 1..16 (overrides config)."),
    ] = None,
    seed_start: Annotated[
        int | None,
        typer.Option("--seed-start", help="Starting seed (overrides config)."),
    ] = None,
    anonymize: Annotated[
        bool,
        typer.Option(help="Redact model name in the rendered report."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the resolved config and exit 0."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Resume an in-progress audit by id (drains pending+stale rows).",
        ),
    ] = None,
    print_stats: Annotated[
        bool,
        typer.Option(
            "--print-stats",
            help="Print aggregate stats (mean/CI/parse-fail/coverage) after the run.",
        ),
    ] = False,
    figures_dir: Annotated[
        Path | None,
        typer.Option(
            "--figures-dir",
            help="Render PDF figures into this directory after the run "
            "(implies --print-stats).",
        ),
    ] = None,
    anonymize_labels: Annotated[
        str | None,
        typer.Option(
            "--anonymize-labels",
            help="Comma-separated anonymisation labels; first element is used "
            "for single-model audits. Falls back to anonymize_label / "
            "'Frontier Model A'.",
        ),
    ] = None,
) -> None:
    """Run a capability audit and emit a PDF report."""
    overrides: dict[str, object] = {
        "model": model,
        "envs": _parse_envs(envs),
        "episodes": episodes,
        "alpha": alpha,
        "output": output,
        "parallel": parallel,
        "seed_start": seed_start,
        # `anonymize` is a flag, default False. Treat False as "not explicitly
        # set" so YAML can opt in. CLI users wanting to FORCE False must edit
        # the YAML or accept the default.
        "anonymize": True if anonymize else None,
    }
    try:
        cfg = load_config(config, overrides)
    except Exception as exc:
        typer.echo(f"vlabs-audit: config error — {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # --figures-dir is meaningless without computed stats; auto-imply.
    # --output triggers the LaTeX render, which needs both stats AND figures.
    if figures_dir is not None and not print_stats:
        print_stats = True
    render_pdf = cfg.output is not None and not dry_run
    if render_pdf:
        print_stats = True  # auto-implied
        # figures_dir gets populated to a tmp dir below if the user
        # didn't pass --figures-dir explicitly.

    if dry_run:
        typer.echo(_format_config(cfg))
        if print_stats:
            typer.echo("")
            typer.echo(
                "(--print-stats: aggregate table will be rendered after a real run; "
                "this dry run does not execute episodes)"
            )
        if figures_dir is not None:
            typer.echo(
                f"(--figures-dir: PDFs will be written to {figures_dir} after a real run)"
            )
        raise typer.Exit(code=0)

    with AuditStore() as store:
        episode_runner = EpisodeRunner(store, parallel=cfg.parallel)
        try:
            if resume:
                audit_id = resume
                processed = episode_runner.resume_audit(audit_id, cfg.model)
                typer.echo(
                    f"vlabs-audit: resumed {audit_id} — {processed} episodes processed."
                )
            else:
                audit_id = episode_runner.run_audit(cfg)
                typer.echo(f"vlabs-audit: audit {audit_id} complete.")
        except ValueError as exc:
            typer.echo(f"vlabs-audit: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        counts = store.counts_by_status(audit_id)
        typer.echo(f"  status: {counts}")

        if print_stats:
            try:
                stats = compute_audit_stats(store, audit_id, alpha=cfg.alpha)
            except ValueError as exc:
                typer.echo(f"vlabs-audit: stats unavailable — {exc}", err=True)
                stats = None
            else:
                typer.echo("")
                typer.echo(format_stats_table(stats))

            # When --anonymize is on, swap the model name on the (in-memory)
            # stats BEFORE rendering. The on-disk audits.db row keeps the
            # real id for reproducibility; the customer artefact does not.
            if stats is not None and cfg.anonymize:
                label = resolve_anonymize_label(
                    explicit_labels=parse_anonymize_labels(anonymize_labels),
                    config_label=cfg.anonymize_label,
                )
                stats = anonymize_audit_stats(stats, label=label)
                typer.echo("")
                typer.echo(f"  anonymisation: model -> {label!r}")

            # Figures land if the user asked for them OR if --output requires
            # them; in the auto path we use a temp dir so the wheel ships
            # nothing to the user's filesystem outside the requested PDF.
            if stats is not None and (figures_dir is not None or render_pdf):
                tmp_figs_ctx: tempfile.TemporaryDirectory[str] | None = None
                if figures_dir is None:
                    tmp_figs_ctx = tempfile.TemporaryDirectory(
                        prefix="vlabs-audit-figs-"
                    )
                    figures_dir_resolved = Path(tmp_figs_ctx.name)
                else:
                    figures_dir_resolved = figures_dir
                try:
                    paths = render_all_figures(stats, figures_dir_resolved)
                except ValueError as exc:
                    typer.echo(
                        f"vlabs-audit: figures unavailable — {exc}", err=True
                    )
                    paths = []
                else:
                    if figures_dir is not None:
                        typer.echo("")
                        typer.echo(
                            f"Wrote {len(paths)} figures to {figures_dir}"
                        )

                if render_pdf and paths:
                    try:
                        pdf_path = render_report(
                            stats, figures_dir_resolved, cfg.output
                        )
                        typer.echo("")
                        typer.echo(f"Report rendered to {pdf_path}")
                    except RuntimeError as exc:
                        typer.echo(
                            f"vlabs-audit: LaTeX render skipped — {exc}",
                            err=True,
                        )

                if tmp_figs_ctx is not None:
                    tmp_figs_ctx.cleanup()
        else:
            typer.echo("  (use --print-stats / --figures-dir / --output to render)")


@app.command()
def version() -> None:
    """Print the vlabs-audit version."""
    typer.echo(f"vlabs-audit {__version__}")


def main() -> int:
    """Entry point used by tests; mirrors the ``vlabs-audit`` console script."""
    try:
        app()
        return 0
    except SystemExit as exc:  # Typer raises SystemExit; surface the exit code
        return int(exc.code) if exc.code is not None else 0


if __name__ == "__main__":
    sys.exit(main())
