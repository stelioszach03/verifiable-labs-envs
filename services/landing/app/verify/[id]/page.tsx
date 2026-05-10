/**
 * Public V-Certified verification page (Phase 31.D).
 *
 * Renders the public-facing verification view for a single attestation
 * keyed by ``public_id``. No auth — anyone with the public_id can hit
 * this page; the API endpoint behind it is per-IP rate-limited.
 *
 * URL pattern: ``/verify/vl-XXXXXXXX`` (8-char Crockford base32 suffix).
 */
export const runtime = "edge";

type AttestationPublicInfo = {
  public_id: string;
  organization: string;
  scope_type: "model" | "deployment" | "organization";
  scope_subject: string;
  tier: "bronze" | "silver" | "gold";
  status:
    | "approved"
    | "revoked"
    | "expired"
    | "draft"
    | "submitted"
    | "under_review"
    | "withdrawn";
  cycle: "annual" | "continuous";
  issued_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
  revocation_reason: string | null;
  cert_serial: string | null;
  certificate_pem: string | null;
  standards_alignment: {
    standards: string[];
    crosswalk_version: string | null;
    framework_versions: Record<string, string>;
  };
};

async function fetchVerify(
  publicId: string,
): Promise<AttestationPublicInfo | { error: string; status: number }> {
  const apiBase =
    process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";
  const r = await fetch(
    `${apiBase}/v1/attestations/verify/${encodeURIComponent(publicId)}`,
    { cache: "no-store" },
  );
  if (!r.ok) {
    return { error: await r.text(), status: r.status };
  }
  return (await r.json()) as AttestationPublicInfo;
}

const TIER_LABEL: Record<string, string> = {
  bronze: "Bronze (annual, self-attested)",
  silver: "Silver (annual, Vlabs-audited)",
  gold: "Gold (continuous, third-party-audited)",
};

const STATUS_COLOR: Record<string, string> = {
  approved: "text-green-600",
  revoked: "text-red-600",
  expired: "text-red-600",
};

export default async function VerifyAttestationPage(props: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await props.params;
  const data = await fetchVerify(id);

  if ("error" in data) {
    return (
      <main className="container-tight py-16">
        <h1 className="text-3xl font-semibold tracking-tight">
          Verification failed
        </h1>
        <p className="mt-2 text-sm text-ink-muted">
          No public V-Certified attestation matches{" "}
          <code className="font-mono">{id}</code>. The attestation may be
          a draft, withdrawn, or the identifier may be mistyped.
        </p>
      </main>
    );
  }

  return (
    <main className="container-tight py-16">
      <header>
        <p className="font-mono text-xs uppercase tracking-wider text-ink-muted">
          V-Certified · {data.public_id}
        </p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">
          {data.organization}
        </h1>
        <p
          className={`mt-2 text-sm uppercase ${
            STATUS_COLOR[data.status] ?? "text-ink-muted"
          }`}
        >
          {data.status}
        </p>
      </header>

      <section className="mt-10 grid gap-x-8 gap-y-3 text-sm md:grid-cols-2">
        <Detail label="Tier" value={TIER_LABEL[data.tier] ?? data.tier} />
        <Detail label="Cycle" value={data.cycle} />
        <Detail label="Scope type" value={data.scope_type} />
        <Detail
          label="Scope subject"
          value={data.scope_subject}
          mono
        />
        <Detail
          label="Issued at"
          value={
            data.issued_at
              ? new Date(data.issued_at).toLocaleString()
              : "—"
          }
        />
        <Detail
          label="Expires at"
          value={
            data.expires_at
              ? new Date(data.expires_at).toLocaleString()
              : "—"
          }
        />
        <Detail
          label="Cert serial"
          value={data.cert_serial ?? "—"}
          mono
        />
        <Detail
          label="Standards"
          value={
            data.standards_alignment.standards.length === 0
              ? "—"
              : data.standards_alignment.standards.join(", ")
          }
          mono
        />
        {data.revoked_at ? (
          <>
            <Detail
              label="Revoked at"
              value={new Date(data.revoked_at).toLocaleString()}
            />
            <Detail
              label="Revocation reason"
              value={data.revocation_reason ?? "—"}
            />
          </>
        ) : null}
      </section>

      {data.certificate_pem ? (
        <section className="mt-12">
          <h2 className="text-xl font-semibold tracking-tight">
            X.509 leaf certificate
          </h2>
          <p className="mt-1 text-sm text-ink-muted">
            Signed by the V-Certified intermediate CA. Verify offline with{" "}
            <code className="font-mono">openssl verify</code> against the
            CA chain at{" "}
            <code className="font-mono">/v1/attestations/crl.pem</code>.
          </p>
          <pre className="mt-4 overflow-x-auto rounded border border-ink/10 bg-ink/5 p-4 text-xs font-mono">
            {data.certificate_pem}
          </pre>
        </section>
      ) : null}
    </main>
  );
}

function Detail({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="grid grid-cols-2 gap-2 border-b border-ink/10 pb-2">
      <dt className="text-ink-muted">{label}</dt>
      <dd className={mono ? "font-mono" : undefined}>{value}</dd>
    </div>
  );
}
