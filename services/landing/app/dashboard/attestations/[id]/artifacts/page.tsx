import Link from "next/link";

import { actUploadArtifact } from "../../actions";

export const runtime = "edge";

const ARTIFACT_KINDS = [
  { id: "training_doc", label: "Training documentation" },
  { id: "audit_report", label: "Audit report" },
  { id: "monitor_record", label: "Monitor record (Phase 28)" },
  { id: "rm_record", label: "Reward model record (Phase 29)" },
  { id: "prm_record", label: "Process reward model record (Phase 30)" },
  { id: "change_mgmt", label: "Change-management trail" },
  { id: "legal_signoff", label: "Legal sign-off" },
  { id: "third_party_audit", label: "Third-party audit attestation" },
];

export default async function UploadArtifactPage(props: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await props.params;

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          Upload evidence artifact
        </h1>
        <Link
          href={`/dashboard/attestations/${id}`}
          className="text-sm text-ink-muted hover:text-ink"
        >
          ← Back to attestation
        </Link>
      </div>
      <p className="mt-1 text-sm text-ink-muted">
        Step 2 of 3 — attach the evidence required by your tier (Bronze:
        training doc + audit report + legal sign-off; Silver adds monitor
        record; Gold adds change-management + at least one of RM / PRM
        records). Files cap at 50 MB after base64 decode. Encrypted blobs
        carry a flag for the auditor.
      </p>

      <form
        action={actUploadArtifact}
        encType="multipart/form-data"
        className="mt-8 grid gap-4 max-w-xl"
      >
        <input type="hidden" name="id" value={id} />

        <label className="grid gap-1 text-sm">
          <span className="font-medium">Artifact kind</span>
          <select
            name="kind"
            defaultValue="training_doc"
            className="rounded border border-ink/20 px-3 py-2"
          >
            {ARTIFACT_KINDS.map((k) => (
              <option key={k.id} value={k.id}>
                {k.label}
              </option>
            ))}
          </select>
        </label>

        <label className="grid gap-1 text-sm">
          <span className="font-medium">File</span>
          <input
            name="file"
            type="file"
            required
            className="rounded border border-ink/20 px-3 py-2"
          />
        </label>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" name="encrypted" />
          <span>This file is client-side encrypted (auditor-only)</span>
        </label>

        <button
          type="submit"
          className="mt-4 rounded bg-ink px-4 py-2 text-sm text-white hover:bg-ink/80"
        >
          Upload artifact
        </button>
      </form>
    </section>
  );
}
