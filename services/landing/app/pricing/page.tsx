import Link from "next/link";

export const runtime = "edge";

export const metadata = {
  title: "Pricing · Verifiable Labs",
  description:
    "Apache-2.0 free SDK + 3 reference environments. Paid hosted reward models, all-envs subscription, V-Certified attestations, and self-hosted enterprise license.",
};

type CtaStyle = "ghost" | "accent" | "outline";

interface Tier {
  id: string;
  name: string;
  cadence: string;
  price: string;
  priceCadence: string;
  blurb: string;
  features: string[];
  cta: { label: string; href: string; style: CtaStyle };
  highlight?: boolean;
}

const TIERS: Tier[] = [
  {
    id: "tier-0",
    name: "Free SDK",
    cadence: "Apache-2.0",
    price: "$0",
    priceCadence: "forever",
    blurb:
      "Production-ready conformal-calibration math + 3 reference environments. No telemetry, no feature gates, no usage cap.",
    features: [
      "vlabs-calibrate (5-line conformal-coverage API)",
      "verifiable-labs CLI + SDK",
      "verifiable-labs-envs runtime",
      "3 reference envs (math-algebra, code-humaneval, sparse-fourier)",
      "Full source on GitHub",
    ],
    cta: {
      label: "pip install verifiable-labs",
      href: "https://pypi.org/project/verifiable-labs/",
      style: "ghost",
    },
  },
  {
    id: "tier-1",
    name: "Hosted RM API",
    cadence: "Pay-as-you-go",
    price: "$0.10",
    priceCadence: "/ 1K API calls",
    blurb:
      "Calibrated reward models served through /v1/reward-models/* and /v1/process-reward-models/* — no GPUs to manage.",
    features: [
      "Outcome + per-step reward scoring",
      "95% conformal coverage on every prediction",
      "100ms p95 latency for batched calls",
      "Free 10K calls/month while in test mode",
      "No subscription, no minimum",
    ],
    cta: { label: "Start free", href: "/sign-up", style: "accent" },
    highlight: true,
  },
  {
    id: "tier-2",
    name: "All-Envs Subscription",
    cadence: "Per-org subscription",
    price: "$499",
    priceCadence: "/ month",
    blurb:
      "All 25 production-grade RL environments (the 22 premium envs beyond the free reference set) — long-context, lodopab CT, MRI knee, sparse Fourier, SQL, code-mini-repo, tool-calling families.",
    features: [
      "25 production-grade envs",
      "Continuous PyPI distribution per env",
      "Bug-fix guarantee within 14 days",
      "Email support, 48h response",
      "Unlimited usage, all envs included as new ones ship",
    ],
    cta: {
      label: "Contact sales",
      href: "mailto:sales@verifiable-labs.com?subject=Tier%202%20%E2%80%94%20All%20envs%20subscription",
      style: "outline",
    },
  },
  {
    id: "tier-3",
    name: "V-Certified Bronze",
    cadence: "Annual, self-attested",
    price: "$4,999",
    priceCadence: "/ year",
    blurb:
      "Public registry attestation that your AI training data + evaluation pipeline meets the V-Certified Bronze rigor bar. Self-attested with a 5-10 day Vlabs review.",
    features: [
      "3 evidence artifacts (training doc, audit report, legal sign-off)",
      "X.509 certificate signed by V-Certified CA",
      "Public registry entry at verify.verifiable-labs.com",
      "Embeddable status badge SVG",
      "Standards crosswalks: ISO 42001, NIST AI RMF, EU AI Act, SOC 2",
    ],
    cta: {
      label: "Start a draft",
      href: "/dashboard/attestations/new",
      style: "outline",
    },
  },
  {
    id: "tier-4",
    name: "V-Certified Silver",
    cadence: "Annual, Vlabs-audited",
    price: "$24,999",
    priceCadence: "/ year",
    blurb:
      "Bronze + Vlabs auditors review the underlying claims (not just artifact shape). For Series B+ AI companies pre-procurement-conversation with regulated buyers.",
    features: [
      "4 evidence artifacts (Bronze + monitor records)",
      "Vlabs auditors review claims, not just artifact shape",
      "Continuous monitoring via Phase 28 monitors",
      "Quarterly check-ins",
      "15-25 business day issuance",
    ],
    cta: {
      label: "Contact sales",
      href: "mailto:sales@verifiable-labs.com?subject=Tier%204%20%E2%80%94%20V-Certified%20Silver",
      style: "outline",
    },
  },
  {
    id: "tier-5",
    name: "V-Certified Gold + Custom Envs",
    cadence: "Annual, third-party-audited",
    price: "$99,999+",
    priceCadence: "/ year",
    blurb:
      "Continuous third-party-audited attestation + 1-3 custom RL environments built to spec by the Verifiable Labs engineering team. For enterprise AI labs and high-stakes deployers.",
    features: [
      "5+ evidence artifacts (Silver + change-mgmt + RM/PRM records)",
      "Independent third-party auditor (V-Certified partner panel)",
      "Continuous recertification (monthly check + annual recert)",
      "1-3 custom envs built to spec",
      "Required for EU AI Act Article 43 conformity assessment input",
      "Dedicated Slack + 4-hour response SLA",
    ],
    cta: {
      label: "Contact sales",
      href: "mailto:sales@verifiable-labs.com?subject=Tier%205%20%E2%80%94%20V-Certified%20Gold%20%2B%20custom%20envs",
      style: "outline",
    },
  },
  {
    id: "tier-6",
    name: "Self-Hosted Enterprise",
    cadence: "On-prem license",
    price: "$250K+",
    priceCadence: "/ year",
    blurb:
      "Full Verifiable Labs stack on-prem. For regulated industries (defense, healthcare, finance) and IP-sensitive frontier labs that refuse to send training data over the wire.",
    features: [
      "API + dashboard + RMs + PRMs + V-Certified on-prem",
      "SSO (SAML/OIDC) integration",
      "Air-gapped Docker images",
      "Reseller / partner-channel terms negotiable",
      "1-hour incident-response SLA",
      "Quarterly on-site engineering review",
    ],
    cta: {
      label: "Contact sales",
      href: "mailto:sales@verifiable-labs.com?subject=Tier%206%20%E2%80%94%20Self-hosted%20enterprise",
      style: "outline",
    },
  },
];

const BTN_CLASS: Record<CtaStyle, string> = {
  accent: "btn-accent w-full",
  ghost: "btn-ghost w-full",
  outline:
    "block w-full rounded border border-ink px-3 py-2 text-center text-sm font-medium hover:bg-ink hover:text-paper",
};

export default function PricingPage() {
  return (
    <main>
      <section className="container-tight py-20">
        <header className="text-center">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-muted">
            Apache-2.0 SDK · paid hosted services
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
            Pricing
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-ink-muted">
            The Verifiable Labs Python SDK is{" "}
            <a
              className="underline"
              href="https://github.com/stelioszach03/verifiable-labs-envs/blob/main/LICENSE"
            >
              Apache-2.0 free forever
            </a>
            . We charge for the operational layer — hosted reward
            models, premium environments, V-Certified attestations,
            and self-hosted enterprise licenses.
          </p>
        </header>

        <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {TIERS.map((t) => (
            <article
              key={t.id}
              className={`card flex flex-col ${
                t.highlight ? "ring-2 ring-accent" : ""
              }`}
            >
              <header>
                <p className="font-mono text-xs uppercase text-ink-muted">
                  {t.cadence}
                </p>
                <h2 className="mt-2 text-xl font-semibold">{t.name}</h2>
              </header>
              <p className="mt-3 text-sm text-ink-muted">{t.blurb}</p>
              <p className="mt-6 flex items-baseline gap-1">
                <span className="text-3xl font-semibold">{t.price}</span>
                <span className="text-sm text-ink-muted">
                  {t.priceCadence}
                </span>
              </p>
              <ul className="mt-5 grid gap-2 text-sm">
                {t.features.map((f) => (
                  <li key={f} className="flex gap-2">
                    <span className="text-accent">✓</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-auto pt-8">
                <Link href={t.cta.href} className={BTN_CLASS[t.cta.style]}>
                  {t.cta.label}
                </Link>
              </div>
            </article>
          ))}
        </div>

        <footer className="mt-16 grid gap-3 text-center text-xs text-ink-muted">
          <p>
            Pricing numbers above are placeholders that firm up once
            the first three Tier 1 + Tier 3 customers close.
          </p>
          <p>
            All paid tiers are in <strong>test mode</strong> until the
            Verifiable Labs Inc. (Delaware C-corp) registration
            completes. Email{" "}
            <a
              className="underline"
              href="mailto:sales@verifiable-labs.com"
            >
              sales@verifiable-labs.com
            </a>{" "}
            to be contacted when live billing opens.
          </p>
          <p>
            Internal strategy reasoning behind the free-vs-paid
            boundary is in{" "}
            <a
              className="underline"
              href="https://github.com/stelioszach03/verifiable-labs-envs/blob/main/docs/BUSINESS_MODEL.md"
            >
              docs/BUSINESS_MODEL.md
            </a>
            .
          </p>
        </footer>
      </section>
    </main>
  );
}
