import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

import {
  type AttestationInfo,
  getAttestation,
  statusColorClass,
} from "@/lib/attestations";

import {
  actRevokeAttestation,
  actSubmitAttestation,
  actWithdrawAttestation,
} from "../actions";

export const runtime = "edge";

async function fetchOne(
  token: string,
  id: string,
): Promise<AttestationInfo | null> {
  try {
    return await getAttestation(token, id);
  } catch {
    return null;
  }
}

export default async function AttestationDetailPage(props: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await props.params;
  const { getToken } = await auth();
  const token = await getToken({ template: "vlabs-api" });
  const att = token ? await fetchOne(token, id) : null;

  if (!att) {
    return (
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">
          Attestation not found
        </h1>
        <p className="mt-2 text-sm text-ink-muted">
          The attestation does not exist or you don&rsquo;t have access.{" "}
          <Link href="/dashboard/attestations" className="underline">
            Back to list
          </Link>
        </p>
      </section>
    );
  }

  const isDraft = att.status === "draft";
  const isApproved = att.status === "approved";
  const isTerminal =
    att.status === "revoked" ||
    att.status === "expired" ||
    att.status === "withdrawn";

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          {att.organization}
        </h1>
        <span
          className={`text-sm uppercase ${statusColorClass(att.status)}`}
        >
          {att.status}
        </span>
      </div>
      <p className="mt-1 font-mono text-xs text-ink-muted">
        {att.public_id} · {att.tier} · {att.scope_type} · {att.cycle}
      </p>

      <dl className="mt-8 grid grid-cols-2 gap-x-6 gap-y-3 text-sm max-w-xl">
        <dt className="text-ink-muted">Scope subject</dt>
        <dd className="font-mono">{att.scope_subject}</dd>

        <dt className="text-ink-muted">Standards</dt>
        <dd className="font-mono">
          {att.standards_alignment.standards.length === 0
            ? "—"
            : att.standards_alignment.standards.join(", ")}
        </dd>

        <dt className="text-ink-muted">Artifacts uploaded</dt>
        <dd>{att.artifact_count}</dd>

        <dt className="text-ink-muted">Issued at</dt>
        <dd className="font-mono">
          {att.issued_at ? new Date(att.issued_at).toLocaleString() : "—"}
        </dd>

        <dt className="text-ink-muted">Expires at</dt>
        <dd className="font-mono">
          {att.expires_at ? new Date(att.expires_at).toLocaleString() : "—"}
        </dd>

        <dt className="text-ink-muted">Cert serial</dt>
        <dd className="font-mono break-all">{att.cert_serial ?? "—"}</dd>

        {att.revoked_at ? (
          <>
            <dt className="text-ink-muted">Revoked at</dt>
            <dd className="font-mono">
              {new Date(att.revoked_at).toLocaleString()}
            </dd>
            <dt className="text-ink-muted">Reason</dt>
            <dd>{att.revocation_reason}</dd>
          </>
        ) : null}
      </dl>

      <div className="mt-10 flex flex-wrap gap-3">
        {isDraft ? (
          <>
            <Link
              href={`/dashboard/attestations/${att.id}/artifacts`}
              className="rounded border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-white"
            >
              Upload artifact
            </Link>
            <form action={actSubmitAttestation}>
              <input type="hidden" name="id" value={att.id} />
              <button
                type="submit"
                className="rounded bg-ink px-3 py-1.5 text-sm text-white hover:bg-ink/80"
              >
                Submit for review
              </button>
            </form>
            <form action={actWithdrawAttestation}>
              <input type="hidden" name="id" value={att.id} />
              <button
                type="submit"
                className="rounded border border-ink/40 px-3 py-1.5 text-sm text-ink-muted hover:text-ink"
              >
                Withdraw
              </button>
            </form>
          </>
        ) : null}
        {isApproved ? (
          <Link
            href={`/dashboard/attestations/${att.id}/renew`}
            className="rounded border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-white"
          >
            Initiate renewal
          </Link>
        ) : null}
        {!isTerminal ? (
          <form
            action={actRevokeAttestation}
            className="flex items-center gap-2"
          >
            <input type="hidden" name="id" value={att.id} />
            <input
              name="reason"
              placeholder="Revocation reason"
              required
              className="rounded border border-ink/20 px-2 py-1 text-sm"
            />
            <button
              type="submit"
              className="rounded border border-red-600 px-3 py-1.5 text-sm text-red-600 hover:bg-red-600 hover:text-white"
            >
              Revoke
            </button>
          </form>
        ) : null}
      </div>
    </section>
  );
}
