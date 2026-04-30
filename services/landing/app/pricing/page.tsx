import Link from "next/link";

const TIERS = [
  {
    name: "Free",
    price: "$0",
    cadence: "forever",
    blurb: "For evaluating fit and small projects.",
    features: [
      "10,000 traces / month",
      "100 requests / minute",
      "All 4 endpoints",
      "Audit history (last 100 evaluations)",
    ],
    cta: { label: "Get started", href: "/sign-up", style: "ghost" as const },
  },
  {
    name: "Pro",
    price: "$99",
    cadence: "/ month",
    blurb: "For production training runs.",
    features: [
      "1,000,000 traces / month",
      "1,000 requests / minute",
      "Metered overage at $1 / 10K traces",
      "Email support, 48h response",
    ],
    cta: { label: "Upgrade", href: "/dashboard/billing", style: "accent" as const },
    highlight: true,
  },
  {
    name: "Team",
    price: "$499",
    cadence: "/ month",
    blurb: "For teams running multiple training pipelines.",
    features: [
      "10,000,000 traces / month",
      "10,000 requests / minute",
      "Metered overage at $0.40 / 10K traces",
      "Priority support, 24h response",
      "Audit retention 12 months",
    ],
    cta: { label: "Upgrade", href: "/dashboard/billing", style: "ghost" as const },
  },
];

export default function PricingPage() {
  return (
    <section className="container-tight py-20">
      <header className="text-center">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Pricing
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-ink-muted">
          Simple per-trace billing. No seats, no per-feature gating.
        </p>
      </header>

      <div className="mt-12 grid gap-6 md:grid-cols-3">
        {TIERS.map((t) => (
          <div
            key={t.name}
            className={`card flex flex-col ${t.highlight ? "ring-2 ring-accent" : ""}`}
          >
            <h2 className="text-xl font-semibold">{t.name}</h2>
            <p className="mt-2 text-sm text-ink-muted">{t.blurb}</p>
            <p className="mt-6 flex items-baseline gap-1">
              <span className="text-4xl font-semibold">{t.price}</span>
              <span className="text-sm text-ink-muted">{t.cadence}</span>
            </p>
            <ul className="mt-6 space-y-2 text-sm">
              {t.features.map((f) => (
                <li key={f} className="flex gap-2">
                  <span className="text-accent">✓</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
            <div className="mt-auto pt-8">
              <Link
                href={t.cta.href}
                className={`btn-${t.cta.style} w-full`}
              >
                {t.cta.label}
              </Link>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-10 text-center text-xs text-ink-muted">
        All paid tiers in test mode until the Verifiable Labs Inc. (Delaware C-corp)
        registration completes. We will email you when live billing opens.
      </p>
    </section>
  );
}
