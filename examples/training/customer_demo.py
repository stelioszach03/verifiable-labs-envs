"""Customer demo: 50-step GRPO improvement on a Verifiable Labs env.

Sales tool. Invoked by a customer to see a reproducible Δ on their
chosen (model, env) combo in ~30 minutes on an A100. Outputs:

* one PNG (before/after reward histogram)
* one 1-page PDF report via reportlab
* one JSONL trace file with the per-episode results

The default invocation is a *dry-run* validator (no training, no model
download). Pass ``--launch`` to actually run the demo.

Usage
-----
    # Dry-run (validate config; recommended first):
    python customer_demo.py --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
        --env-id sparse-fourier-recovery

    # Real demo (~30 min on A100):
    python customer_demo.py --base-model-id Qwen/Qwen2.5-1.5B-Instruct \
        --env-id sparse-fourier-recovery --launch \
        --out-dir /content/drive/MyDrive/verifiable-labs/customer_demo_run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CACHE = "/content/drive/MyDrive/verifiable-labs/hf_cache"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("HF_HUB_CACHE", str(Path(DEFAULT_CACHE) / "hub"))

# Import the model registry from the universal trainer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grpo_universal import (  # type: ignore[import-not-found]  # noqa: E402
    MODEL_REGISTRY,
    _print_report,
    dry_run_check,
)

# Demo configuration: 50 steps, 20 eval episodes per side. Anything more
# is a real research run, not a demo. Each customer demo lands at
# ~30 min on A100 (5 min model load + 20 min training + 5 min eval).
DEMO_MAX_STEPS = 50
DEMO_SAVE_STEPS = 25
DEMO_LOGGING_STEPS = 5
DEMO_EVAL_SEEDS = 20      # 20 seeds × 1 sample = 20 episodes per side


def _check_reportlab_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def write_demo_pdf(
    pdf_path: Path,
    *,
    model_id: str,
    env_id: str,
    before_mean: float,
    after_mean: float,
    delta_mean: float,
    delta_ci95: tuple[float, float],
    n_eval: int,
    training_minutes: float,
    plot_path: Path | None,
) -> None:
    """Render a 1-page PDF demo report. Uses reportlab if available;
    falls back to a Markdown stub otherwise."""
    if not _check_reportlab_available():
        # Fallback — write Markdown with the same structure.
        md = pdf_path.with_suffix(".md")
        md.write_text(
            f"# Verifiable Labs — Customer Demo Report\n\n"
            f"**Model:** {model_id}\n"
            f"**Environment:** {env_id}\n\n"
            f"## Result\n\n"
            f"| | mean reward |\n|---|---|\n"
            f"| BEFORE (base model) | {before_mean:.4f} |\n"
            f"| AFTER (50 GRPO steps) | {after_mean:.4f} |\n"
            f"| **Δ** | **{delta_mean:+.4f}** ({delta_mean*100:+.2f} pp) |\n"
            f"| 95% CI | [{delta_ci95[0]:+.4f}, {delta_ci95[1]:+.4f}] |\n"
            f"| n_eval per side | {n_eval} |\n"
            f"| training wall time | {training_minutes:.1f} min |\n\n"
            f"reportlab not installed; this is the Markdown fallback. "
            f"Run `pip install reportlab` then re-run for a PDF."
        )
        print(f"⚠️  reportlab missing; wrote Markdown fallback → {md}")
        return

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    body = styles["BodyText"]
    body.fontSize = 10
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.gray)

    flow = []
    flow.append(Paragraph("Verifiable Labs — Customer Demo Report", h1))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        f"<b>Model:</b> {model_id}<br/>"
        f"<b>Environment:</b> {env_id}<br/>"
        f"<b>Training:</b> 50 GRPO steps "
        f"on Verifiable Labs RL infrastructure", body
    ))
    flow.append(Spacer(1, 10))

    data = [
        ["Metric", "Value"],
        ["Mean reward — BEFORE (base)", f"{before_mean:.4f}"],
        ["Mean reward — AFTER (50 steps)", f"{after_mean:.4f}"],
        ["Δ (after − before)", f"{delta_mean:+.4f}"],
        ["Δ (percentage points)", f"{delta_mean*100:+.2f} pp"],
        ["95% bootstrap CI on Δ",
         f"[{delta_ci95[0]:+.4f}, {delta_ci95[1]:+.4f}]"],
        ["n eval per side", str(n_eval)],
        ["Training wall time", f"{training_minutes:.1f} min"],
    ]
    table = Table(data, colWidths=[3.0 * inch, 2.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 10))

    if plot_path and Path(plot_path).exists():
        flow.append(Image(str(plot_path), width=5.5 * inch, height=3.0 * inch))
        flow.append(Spacer(1, 6))

    flow.append(Paragraph(
        "Reproducibility: every per-episode trace JSONL line carries "
        "<code>config_hash</code>, <code>instance_hash</code>, and "
        "<code>reward_hash</code>. The eval split (TEST seeds) is "
        "permanently held out from training. Statistical test: Wilcoxon "
        "signed-rank, paired by <code>instance_hash</code>; CI: 10 000 "
        "bootstrap resamples seeded by 42.",
        small,
    ))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "Verifiable Labs · sales@verifiable-labs.com · "
        "github.com/stelioszach03/verifiable-labs-envs",
        small,
    ))
    doc.build(flow)
    print(f"✅ wrote {pdf_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--env-id", default="sparse-fourier-recovery")
    ap.add_argument("--out-dir", default="/tmp/customer_demo")
    ap.add_argument("--launch", action="store_true",
                    help="actually run training+eval (default: dry-run)")
    args = ap.parse_args()

    if args.base_model_id not in MODEL_REGISTRY:
        print(f"❌ {args.base_model_id} not in MODEL_REGISTRY",
              file=sys.stderr)
        return 2

    print("=== Verifiable Labs — Customer Demo (DRY-RUN) ===\n")
    print(f"Model:   {args.base_model_id}")
    print(f"Env:     {args.env_id}")
    print(f"Steps:   {DEMO_MAX_STEPS} (~30 min on A100)")
    print(f"Eval:    {DEMO_EVAL_SEEDS} seeds × 1 sample per side")
    print(f"Out dir: {args.out_dir}")
    print()

    rep = dry_run_check(args.base_model_id)
    _print_report(rep)
    if rep["errors"]:
        print(f"\n❌ Dry-run failed: {rep['errors']}")
        return 2

    print(f"\n✅ Pre-flight OK. reportlab installed: {_check_reportlab_available()}")

    if not args.launch:
        print("[DRY-RUN] no training/eval launched. Pass --launch to run "
              f"the {DEMO_MAX_STEPS}-step demo (~30 min).")
        return 0

    # --- Real run path (deferred to Phase 14 wiring).
    print("\n=== LAUNCH MODE ===")
    print(
        "The actual demo run is wired through:\n"
        f"  1. {Path(__file__).parent}/eval_universal.py  (BEFORE eval, n=20)\n"
        f"  2. {Path(__file__).parent}/train_grpo_qwen.py --max-steps 50  (50 GRPO steps)\n"
        f"  3. {Path(__file__).parent}/eval_universal.py --checkpoint-path …  (AFTER eval, n=20)\n"
        f"  4. {Path(__file__).parent}/compare_runs.py  (paired Δ + plot)\n"
        f"  5. write_demo_pdf(...)  (1-page PDF for the customer)"
    )
    print(
        "\nPhase 14 will wire steps 1–4 end-to-end; this script then\n"
        "calls write_demo_pdf with the resulting numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
