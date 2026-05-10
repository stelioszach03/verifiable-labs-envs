import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

import {
  type AttestationList,
  listAttestations,
  statusColorClass,
} from "@/lib/attestations";

export const runtime = "edge";

async function fetchAttestations(token: string): Promise<AttestationList> {
  try {
    return await listAttestations(token);
  } catch {
    return { items: [], total: 0, limit: 0, offset: 0 };
  }
}

export default async function AttestationsListPage() {
  const { getToken } = await auth();
  const token = await getToken({ template: "vlabs-api" });
  const data = token
    ? await fetchAttestations(token)
    : { items: [], total: 0, limit: 0, offset: 0 };

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          V-Certified attestations
        </h1>
        <Link
          href="/dashboard/attestations/new"
          className="rounded border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-white"
        >
          New attestation
        </Link>
      </div>
      <p className="mt-1 text-sm text-ink-muted">
        Programme certificates for AI training data quality. Bronze (annual,
        self-attested) → Silver (annual, Vlabs-audited) → Gold (continuous,
        third-party-audited). Public verification at{" "}
        <code className="font-mono text-xs">verify.verifiable-labs.com</code>{" "}
        once approved.
      </p>

      {data.items.length === 0 ? (
        <p className="mt-8 text-sm text-ink-muted">
          No attestations yet. Click <em>New attestation</em> above to start
          a draft, or call{" "}
          <code className="font-mono text-xs">POST /v1/attestations</code>{" "}
          directly.
        </p>
      ) : (
        <div className="mt-8 grid gap-4">
          {data.items.map((a) => (
            <Link
              key={a.id}
              href={`/dashboard/attestations/${a.id}`}
              className="card block hover:border-ink"
            >
              <div className="flex items-baseline justify-between">
                <h2 className="text-lg font-semibold">{a.organization}</h2>
                <span
                  className={`text-xs uppercase ${statusColorClass(a.status)}`}
                >
                  {a.status}
                </span>
              </div>
              <p className="mt-1 font-mono text-xs text-ink-muted">
                {a.public_id} · {a.tier} · {a.scope_type} · {a.cycle}
              </p>
              <p className="mt-1 text-xs text-ink-muted">
                {a.scope_subject}
              </p>
              {a.expires_at ? (
                <p className="mt-2 text-xs text-ink-muted">
                  Expires: {new Date(a.expires_at).toLocaleDateString()}
                </p>
              ) : null}
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}
