import Link from "next/link";

export const runtime = "edge";

export const metadata = {
  title: "V-Certified · Verifiable Labs",
  description:
    "V-Certified is the third-party attestation programme for AI training data quality. Bronze (annual self-attested), Silver (Vlabs-audited), Gold (third-party audited) tiers with cryptographically-signed X.509 certificates and a public verification registry.",
};

const TIERS = [
  {
    name: "Bronze",
    color: "border-amber-700",
    cadence: "Annual, self-attested",
    blurb:
      "Customer self-attests with three artifacts: training documentation, audit report, legal sign-off.",
    features: [
      "Annual recertification",
      "3 evidence artifacts (50 MB each)",
      "Public verification registry",
      "Embeddable status badge",
      "Standards crosswalks: ISO 42001, NIST AI RMF, EU AI Act, SOC 2",
    ],
    price: "$2,500",
    priceCadence: "/ year",
  },
  {
    name: "Silver",
    color: "border-zinc-400",
    cadence: "Annual, Vlabs-audited",
    blurb:
      "Verifiable Labs auditors review the customer's evidence + monitor records before issuance.",
    features: [
      "Annual recertification + quarterly check-ins",
      "4 evidence artifacts (Bronze + monitor records)",
      "Vlabs internal review board",
      "Continuous monitoring via Phase 28 monitors",
      "All standards crosswalks",
    ],
    price: "$15,000",
    priceCadence: "/ year",
    highlight: true,
  },
  {
    name: "Gold",
    color: "border-yellow-600",
    cadence: "Continuous, third-party audited",
    blurb:
      "Independent auditor (your choice from our partners) attests on a continuous cycle.",
    features: [
      "Continuous recertification (monthly check + annual recert)",
      "5+ evidence artifacts (Silver + change-mgmt + RM/PRM records)",
      "Third-party audit + Vlabs internal review",
      "Required for EU AI Act conformity assessments",
      "Custom crosswalk extensions on request",
    ],
    price: "$45,000+",
    priceCadence: "/ year",
  },
];

const CONTROLS = [
  {
    id: "VC-1.1",
    title: "AI training-data provenance",
    description:
      "Documented data lineage, lawful-basis records, third-party data attestations.",
  },
  {
    id: "VC-2.1",
    title: "Model evaluation rigor",
    description:
      "Reward / process-reward records, monitor evidence, conformal calibration.",
  },
  {
    id: "VC-3.1",
    title: "Change management trail",
    description:
      "Versioned change log of training data + model updates + deployment changes.",
  },
  {
    id: "VC-4.1",
    title: "Legal accountability",
    description: "Customer legal sign-off + indemnification posture.",
  },
  {
    id: "VC-5.1",
    title: "Programme governance",
    description:
      "Organisational accountability for AI risk management (NIST AI RMF GOVERN).",
  },
  {
    id: "VC-6.1",
    title: "Conformity assessment",
    description: "Third-party conformity assessment artifacts (EU AI Act).",
  },
];

export default function VCertifiedPage() {
  return (
    <main>
      <section className="container-tight py-24">
        <header className="text-center">
          <p className="font-mono text-xs uppercase tracking-wider text-ink-muted">
            Verifiable Labs · Programme launch
          </p>
          <h1 className="mt-4 text-5xl font-semibold tracking-tight sm:text-6xl">
            V-Certified
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-ink-muted">
            A third-party attestation programme for AI training data
            quality. Bronze, Silver, and Gold tiers issue
            cryptographically-signed X.509 certificates verifiable in
            seconds via{" "}
            <code className="font-mono text-sm">
              verify.verifiable-labs.com
            </code>
            .
          </p>
          <div className="mt-10 flex justify-center gap-4">
            <Link
              href="/dashboard/attestations/new"
              className="rounded bg-ink px-5 py-2.5 text-sm font-medium text-white hover:bg-ink/80"
            >
              Start a draft attestation
            </Link>
            <Link
              href="/v1/standards"
              className="rounded border border-ink/40 px-5 py-2.5 text-sm font-medium hover:border-ink"
            >
              View crosswalks
            </Link>
          </div>
        </header>
      </section>

      <section className="container-tight py-16">
        <h2 className="text-3xl font-semibold tracking-tight">
          How it works
        </h2>
        <ol className="mt-8 grid gap-6 md:grid-cols-2">
          <li className="card">
            <h3 className="text-lg font-semibold">1. Draft</h3>
            <p className="mt-2 text-sm text-ink-muted">
              Pick a tier, declare scope, request standards alignment.
              The programme returns a draft attestation with a
              short-form public ID (vl-XXXXXXXX).
            </p>
          </li>
          <li className="card">
            <h3 className="text-lg font-semibold">2. Upload evidence</h3>
            <p className="mt-2 text-sm text-ink-muted">
              Attach 3-6 artifacts depending on tier. SHA-256 hashed at
              upload; sensitive blobs can be client-side encrypted.
              50 MB cap per file.
            </p>
          </li>
          <li className="card">
            <h3 className="text-lg font-semibold">3. Audit decision</h3>
            <p className="mt-2 text-sm text-ink-muted">
              The review board records an approve / reject /
              request-more decision. Approval issues an X.509 leaf cert
              signed by the V-Certified CA.
            </p>
          </li>
          <li className="card">
            <h3 className="text-lg font-semibold">4. Public verification</h3>
            <p className="mt-2 text-sm text-ink-muted">
              The cert + attestation surface publicly at{" "}
              <code className="font-mono text-xs">
                /v1/attestations/verify/{"{public_id}"}
              </code>
              . Verifiers chain to the CA + check the CRL for
              revocation status.
            </p>
          </li>
        </ol>
      </section>

      <section className="container-tight py-16">
        <h2 className="text-3xl font-semibold tracking-tight">Tiers</h2>
        <p className="mt-2 text-sm text-ink-muted">
          Higher tiers carry more rigorous evidence requirements + more
          frequent re-certification.
        </p>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {TIERS.map((t) => (
            <div
              key={t.name}
              className={`card border-2 ${t.color} ${
                t.highlight ? "shadow-lg" : ""
              }`}
            >
              <h3 className="text-2xl font-semibold">{t.name}</h3>
              <p className="mt-1 text-sm text-ink-muted">{t.cadence}</p>
              <p className="mt-4 text-sm">{t.blurb}</p>
              <p className="mt-6 text-3xl font-semibold">
                {t.price}
                <span className="text-sm text-ink-muted">
                  {t.priceCadence}
                </span>
              </p>
              <ul className="mt-4 grid gap-2 text-sm">
                {t.features.map((f) => (
                  <li key={f} className="flex gap-2">
                    <span className="text-ink-muted">→</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="container-tight py-16">
        <h2 className="text-3xl font-semibold tracking-tight">
          Control set
        </h2>
        <p className="mt-2 text-sm text-ink-muted">
          The V-Certified controls map onto ISO/IEC 42001, NIST AI RMF,
          EU AI Act, and SOC 2 Trust Services Criteria. Full crosswalks
          at{" "}
          <code className="font-mono text-xs">/v1/standards/{"{framework}"}</code>
          .
        </p>
        <dl className="mt-10 grid gap-4 md:grid-cols-2">
          {CONTROLS.map((c) => (
            <div key={c.id} className="card">
              <dt>
                <span className="font-mono text-xs text-ink-muted">
                  {c.id}
                </span>
                <span className="ml-2 text-lg font-semibold">
                  {c.title}
                </span>
              </dt>
              <dd className="mt-2 text-sm text-ink-muted">
                {c.description}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="container-tight py-16">
        <h2 className="text-3xl font-semibold tracking-tight">
          Cryptographic verification
        </h2>
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <p className="text-sm text-ink-muted">
            Every approved attestation receives a unique X.509 leaf
            certificate signed by the V-Certified CA. The CN encodes
            the public_id; the OU encodes the cert serial. Verifiers
            chain the leaf to the CA + check the publicly-served CRL
            for revocation.
          </p>
          <pre className="overflow-x-auto rounded bg-ink/5 p-4 text-xs font-mono">
{`# Verify a V-Certified attestation
$ curl https://api.verifiable-labs.com/v1/attestations/verify/vl-ABCD1234 \\
  | jq -r .certificate_pem > leaf.pem
$ curl https://api.verifiable-labs.com/v1/attestations/crl.pem > crl.pem
$ openssl verify -CRLfile crl.pem -crl_check leaf.pem
leaf.pem: OK`}
          </pre>
        </div>
      </section>

      <section className="container-tight py-24 text-center">
        <h2 className="text-3xl font-semibold tracking-tight">
          Ready to get certified?
        </h2>
        <p className="mt-2 text-sm text-ink-muted">
          Email{" "}
          <a
            href="mailto:v-certified@verifiable-labs.com"
            className="underline"
          >
            v-certified@verifiable-labs.com
          </a>{" "}
          to schedule an onboarding call, or start a Bronze attestation
          draft directly from your dashboard.
        </p>
        <div className="mt-8 flex justify-center gap-4">
          <Link
            href="/dashboard/attestations/new"
            className="rounded bg-ink px-5 py-2.5 text-sm font-medium text-white hover:bg-ink/80"
          >
            Start a draft attestation
          </Link>
        </div>
      </section>
    </main>
  );
}
