import Link from "next/link";

export const runtime = "edge";

const FEATURES = [
  {
    title: "Drop-in replacement",
    body: "Wrap any Python reward function with vc.calibrate(...). Returns a callable that emits reward + conformal interval per call.",
  },
  {
    title: "Provable (1−α) coverage",
    body: "Split-conformal prediction (Lei et al., 2018). Marginal coverage guaranteed under exchangeability.",
  },
  {
    title: "Hosted or self-host",
    body: "pip install vlabs-calibrate to run locally. Or call the hosted API from production for usage metering and audit history.",
  },
];

export default function LandingPage() {
  return (
    <>
      <section className="relative overflow-hidden">
        <div className="container-tight pt-24 pb-20 text-center">
          <p className="mb-6 inline-flex items-center gap-2 rounded-full border border-ink/10 bg-white px-3 py-1 text-xs text-ink-muted">
            <span className="h-2 w-2 rounded-full bg-accent" />
            v0.1.0a1 — alpha
          </p>
          <h1 className="text-balance text-5xl font-semibold tracking-tight sm:text-6xl">
            Calibration infrastructure for{" "}
            <span className="text-accent">RL training</span>.
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-ink-muted">
            Every RL training run today ships uncalibrated rewards. Verifiable Labs
            wraps any reward function with provable conformal coverage in five lines.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-3">
            <Link href="/sign-up" className="btn-accent">
              Get started — free 10K traces / month
            </Link>
            <Link href="/pricing" className="btn-ghost">
              See pricing
            </Link>
          </div>
          <pre className="mx-auto mt-12 max-w-2xl rounded-xl bg-ink p-6 text-left text-sm text-paper shadow-lg">
{`import vlabs_calibrate as vc

calibrated = vc.calibrate(my_reward, traces, alpha=0.1)
result = calibrated(prompt=..., completion=..., sigma=0.5)
# → reward, interval, target_coverage`}
          </pre>
        </div>
      </section>

      <section className="container-tight grid gap-6 py-20 sm:grid-cols-3">
        {FEATURES.map((f) => (
          <div key={f.title} className="card">
            <h3 className="text-lg font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm text-ink-muted">{f.body}</p>
          </div>
        ))}
      </section>

      <section className="container-tight py-12">
        <div className="card flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-semibold">
              Ready to calibrate your reward model?
            </h2>
            <p className="mt-2 text-ink-muted">
              Free tier covers 10,000 traces/month. No credit card required.
            </p>
          </div>
          <Link href="/sign-up" className="btn-accent shrink-0">
            Create your API key
          </Link>
        </div>
      </section>
    </>
  );
}
