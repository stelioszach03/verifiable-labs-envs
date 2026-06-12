"""vlabs-prm-eval — CLI entrypoint.

Currently exposes a single subcommand ``gate`` which evaluates the
machine-verified 7-condition self-improvement gate from
``formal/VerifiableLabsFormal/SelfImprovementGate.lean`` against two
eval cards (old and new) and exits 0 on ACCEPT, 1 on REJECT.

See ``vlabs-prm-eval gate --help`` for the full mapping and tolerance
flags. Additional subcommands (``card``, ``aggregate``, ``promote``)
are planned in subsequent phases but intentionally not implemented
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from verifiable_labs_envs.formal_spec.clean_promotion_gate import CleanTolerances
from verifiable_labs_envs.formal_spec.gate import Tolerances

from .clean_gate import evaluate_clean_gate
from .gate import evaluate_gate

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Verifiable Labs process-reward-model eval card tooling.",
)


@app.callback()
def _root() -> None:
    """Root callback — forces typer multi-command mode so ``gate`` is a
    real subcommand rather than the implicit single command.

    Additional subcommands (``card``, ``aggregate``, ``promote``) will
    register here in subsequent phases without breaking the existing
    ``vlabs-prm-eval gate ...`` invocation.
    """


@app.command(
    "gate",
    help=(
        "Evaluate the 7-condition checkpoint promotion gate "
        "(SelfImprovementGate.lean) against two eval cards. "
        "Exit 0 on ACCEPT, 1 on REJECT."
    ),
)
def gate(
    old: Annotated[Path, typer.Option(
        "--old", "-O", help="Path to the previous-checkpoint eval card (JSON).",
        exists=True, dir_okay=False, readable=True,
    )],
    new: Annotated[Path, typer.Option(
        "--new", "-N", help="Path to the candidate-checkpoint eval card (JSON).",
        exists=True, dir_okay=False, readable=True,
    )],
    metrics_map: Annotated[Path | None, typer.Option(
        "--metrics-map",
        help="Optional JSON file overriding DEFAULT_METRICS_MAP "
             "(vgs_weights, regression_rules).",
        dir_okay=False, readable=True,
    )] = None,
    tau: Annotated[float, typer.Option(
        "--tau", help="Minimum VGS gain required for an accept.", min=0.0,
    )] = 0.01,
    eps_h: Annotated[float, typer.Option(
        "--eps-h", help="Allowed increase in hack_risk.", min=0.0,
    )] = 0.02,
    eps_c: Annotated[float, typer.Option(
        "--eps-c", help="Allowed drop in calibration (empirical coverage).", min=0.0,
    )] = 0.02,
    eps_o: Annotated[float, typer.Option(
        "--eps-o", help="Allowed drop in OOD (mean held-out env metric).", min=0.0,
    )] = 0.02,
    eps_k: Annotated[float, typer.Option(
        "--eps-k", help="Allowed cost increase ($/audit).", min=0.0,
    )] = 5.0,
    eps_l: Annotated[float, typer.Option(
        "--eps-l", help="Allowed latency increase (seconds/audit).", min=0.0,
    )] = 0.5,
    cost_old: Annotated[float | None, typer.Option(
        "--cost-old", help="Override old-card cost if not present in JSON.",
    )] = None,
    cost_new: Annotated[float | None, typer.Option(
        "--cost-new", help="Override new-card cost if not present in JSON.",
    )] = None,
    latency_old: Annotated[float | None, typer.Option(
        "--latency-old", help="Override old-card latency if not present in JSON.",
    )] = None,
    latency_new: Annotated[float | None, typer.Option(
        "--latency-new", help="Override new-card latency if not present in JSON.",
    )] = None,
) -> None:
    tol = Tolerances(tau=tau, eps_h=eps_h, eps_c=eps_c, eps_o=eps_o, eps_k=eps_k, eps_l=eps_l)
    decision, old_m, new_m, warnings = evaluate_gate(
        old_card_path=old,
        new_card_path=new,
        tol=tol,
        metrics_map_path=metrics_map,
        cost_old=cost_old,
        cost_new=cost_new,
        latency_old=latency_old,
        latency_new=latency_new,
    )

    for w in warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW, err=True)

    header = "ACCEPT" if decision.accepted else "REJECT"
    color = typer.colors.GREEN if decision.accepted else typer.colors.RED
    typer.secho(f"== {header} ==", fg=color, bold=True)
    typer.echo("")
    typer.echo(f"{'condition':<32} {'old':>10} {'new':>10} {'budget':>10} {'OK'}")
    typer.echo(f"{'-' * 32} {'-' * 10} {'-' * 10} {'-' * 10} {'--'}")

    def row(name: str, old_val: object, new_val: object, budget: object, ok: bool) -> None:
        mark = "OK" if ok else "!!"

        def fmt(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        typer.echo(
            f"{name:<32} {fmt(old_val):>10} {fmt(new_val):>10} {fmt(budget):>10}  {mark}"
        )

    row("vgs_gain >= tau", old_m.vgs, new_m.vgs, tol.tau, new_m.vgs >= old_m.vgs + tol.tau)
    row("hack_risk <= eps_h", old_m.hack_risk, new_m.hack_risk, tol.eps_h,
        new_m.hack_risk <= old_m.hack_risk + tol.eps_h)
    row("calibration >= -eps_c", old_m.calibration, new_m.calibration, tol.eps_c,
        new_m.calibration >= old_m.calibration - tol.eps_c)
    row("ood >= -eps_o", old_m.ood, new_m.ood, tol.eps_o,
        new_m.ood >= old_m.ood - tol.eps_o)
    row("cost <= eps_k", old_m.cost, new_m.cost, tol.eps_k,
        new_m.cost <= old_m.cost + tol.eps_k)
    row("latency <= eps_l", old_m.latency, new_m.latency, tol.eps_l,
        new_m.latency <= old_m.latency + tol.eps_l)
    row("regression flag", old_m.regression, new_m.regression, "False",
        not new_m.regression)

    if decision.reasons:
        typer.echo("")
        typer.secho("Reasons:", fg=typer.colors.RED, bold=True)
        for r in decision.reasons:
            typer.echo(f"  - {r}")

    raise typer.Exit(code=0 if decision.accepted else 1)


@app.command(
    "clean-gate",
    help=(
        "Evaluate the 8-condition contamination-resistant clean promotion gate "
        "(CleanPromotionGate.lean) against two eval cards. "
        "Exit 0 on ACCEPT, 1 on REJECT."
    ),
)
def clean_gate(
    old: Annotated[Path, typer.Option(
        "--old", "-O", help="Path to the previous-checkpoint eval card (JSON).",
        exists=True, dir_okay=False, readable=True,
    )],
    new: Annotated[Path, typer.Option(
        "--new", "-N", help="Path to the candidate-checkpoint eval card (JSON).",
        exists=True, dir_okay=False, readable=True,
    )],
    tau: Annotated[float, typer.Option(
        "--tau", help="Minimum clean_vgs gain required for an accept.", min=0.0,
    )] = 0.01,
    eps_h: Annotated[float, typer.Option(
        "--eps-h", help="Allowed increase in hack_risk.", min=0.0,
    )] = 0.02,
    eps_c: Annotated[float, typer.Option(
        "--eps-c", help="Allowed drop in calibration.", min=0.0,
    )] = 0.02,
    eps_o: Annotated[float, typer.Option(
        "--eps-o", help="Allowed drop in ood_score.", min=0.0,
    )] = 0.02,
    eps_d: Annotated[float, typer.Option(
        "--eps-d", help="Allowed increase in contamination risk (dcr).", min=0.0,
    )] = 0.02,
    eps_k: Annotated[float, typer.Option(
        "--eps-k", help="Allowed cost increase ($/audit).", min=0.0,
    )] = 5.0,
    eps_l: Annotated[float, typer.Option(
        "--eps-l", help="Allowed latency increase (seconds/audit).", min=0.0,
    )] = 0.5,
    beta: Annotated[float, typer.Option(
        "--beta",
        help="Contamination penalty weight used when clean_vgs must be computed "
             "from raw_vgs and dcr.",
        min=0.0,
    )] = 0.5,
    metrics_map: Annotated[Path | None, typer.Option(
        "--metrics-map",
        help="Optional JSON file overriding DEFAULT_METRICS_MAP "
             "(field aliases / fallbacks).",
        dir_okay=False, readable=True,
    )] = None,
    cost_old: Annotated[float | None, typer.Option(
        "--cost-old", help="Override old-card cost if not present in JSON.",
    )] = None,
    cost_new: Annotated[float | None, typer.Option(
        "--cost-new", help="Override new-card cost if not present in JSON.",
    )] = None,
    latency_old: Annotated[float | None, typer.Option(
        "--latency-old", help="Override old-card latency if not present in JSON.",
    )] = None,
    latency_new: Annotated[float | None, typer.Option(
        "--latency-new", help="Override new-card latency if not present in JSON.",
    )] = None,
) -> None:
    tol = CleanTolerances(
        tau=tau, eps_h=eps_h, eps_c=eps_c, eps_o=eps_o, eps_d=eps_d,
        eps_k=eps_k, eps_l=eps_l,
    )
    decision, old_m, new_m, warnings = evaluate_clean_gate(
        old_card_path=old,
        new_card_path=new,
        tol=tol,
        beta=beta,
        metrics_map_path=metrics_map,
        cost_old=cost_old,
        cost_new=cost_new,
        latency_old=latency_old,
        latency_new=latency_new,
    )

    for w in warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW, err=True)

    header = "ACCEPT" if decision.accepted else "REJECT"
    color = typer.colors.GREEN if decision.accepted else typer.colors.RED
    typer.secho(f"== {header} ==", fg=color, bold=True)
    typer.echo("")
    typer.echo(f"{'condition':<32} {'old':>10} {'new':>10} {'budget':>10} {'OK'}")
    typer.echo(f"{'-' * 32} {'-' * 10} {'-' * 10} {'-' * 10} {'--'}")

    def row(name: str, old_val: object, new_val: object, budget: object, ok: bool) -> None:
        mark = "OK" if ok else "!!"

        def fmt(v: object) -> str:
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        typer.echo(
            f"{name:<32} {fmt(old_val):>10} {fmt(new_val):>10} {fmt(budget):>10}  {mark}"
        )

    row("clean_vgs >= +tau", old_m.clean_vgs, new_m.clean_vgs, tol.tau,
        new_m.clean_vgs >= old_m.clean_vgs + tol.tau)
    row("hack_risk <= +eps_h", old_m.hack_risk, new_m.hack_risk, tol.eps_h,
        new_m.hack_risk <= old_m.hack_risk + tol.eps_h)
    row("calibration >= -eps_c", old_m.calibration, new_m.calibration, tol.eps_c,
        new_m.calibration >= old_m.calibration - tol.eps_c)
    row("ood_score >= -eps_o", old_m.ood_score, new_m.ood_score, tol.eps_o,
        new_m.ood_score >= old_m.ood_score - tol.eps_o)
    row("dcr <= +eps_d", old_m.dcr, new_m.dcr, tol.eps_d,
        new_m.dcr <= old_m.dcr + tol.eps_d)
    row("cost <= +eps_k", old_m.cost, new_m.cost, tol.eps_k,
        new_m.cost <= old_m.cost + tol.eps_k)
    row("latency <= +eps_l", old_m.latency, new_m.latency, tol.eps_l,
        new_m.latency <= old_m.latency + tol.eps_l)
    row("regression flag", old_m.regression, new_m.regression, "False",
        not new_m.regression)

    if decision.reasons:
        typer.echo("")
        typer.secho("Reasons:", fg=typer.colors.RED, bold=True)
        for r in decision.reasons:
            typer.echo(f"  - {r}")

    raise typer.Exit(code=0 if decision.accepted else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
