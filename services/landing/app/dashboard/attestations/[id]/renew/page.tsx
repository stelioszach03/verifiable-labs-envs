import Link from "next/link";

import { actInitiateRenewal } from "../../actions";

export const runtime = "edge";

export default async function RenewAttestationPage(props: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await props.params;
  // Pre-generate an idempotency key per render so the form is safe to
  // resubmit. Server action also generates one if the field is empty.
  const idempotencyKey = crypto.randomUUID();

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          Initiate renewal cycle
        </h1>
        <Link
          href={`/dashboard/attestations/${id}`}
          className="text-sm text-ink-muted hover:text-ink"
        >
          ← Back to attestation
        </Link>
      </div>
      <p className="mt-1 text-sm text-ink-muted">
        Renewal opens a new audit cycle for this attestation. Bronze and
        Silver renew annually; Gold renews continuously after change
        events. The idempotency key below protects against accidental
        double-submits within a 24-hour window.
      </p>

      <form
        action={actInitiateRenewal}
        className="mt-8 grid gap-4 max-w-xl"
      >
        <input type="hidden" name="id" value={id} />
        <label className="grid gap-1 text-sm">
          <span className="font-medium">Idempotency key</span>
          <input
            name="idempotency_key"
            defaultValue={idempotencyKey}
            className="rounded border border-ink/20 px-3 py-2 font-mono text-xs"
            readOnly
          />
        </label>
        <button
          type="submit"
          className="mt-4 rounded bg-ink px-4 py-2 text-sm text-white hover:bg-ink/80"
        >
          Open renewal
        </button>
      </form>
    </section>
  );
}
