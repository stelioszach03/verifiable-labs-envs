import Link from "next/link";

import { actCreateAttestation } from "../actions";

export const runtime = "edge";

const STANDARDS = [
  { id: "iso_42001", label: "ISO/IEC 42001 (AI management system)" },
  { id: "nist_ai_rmf", label: "NIST AI Risk Management Framework" },
  { id: "eu_ai_act", label: "EU AI Act (high-risk system controls)" },
  { id: "soc2", label: "SOC 2 Trust Services Criteria" },
];

export default function NewAttestationPage() {
  return (
    <section>
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          New attestation
        </h1>
        <Link
          href="/dashboard/attestations"
          className="text-sm text-ink-muted hover:text-ink"
        >
          ← Back to list
        </Link>
      </div>
      <p className="mt-1 text-sm text-ink-muted">
        Step 1 of 3 — programme metadata. After creation you upload the
        evidence artifacts (training docs, audit reports, monitoring +
        reward records, change-management trail, legal sign-off), then
        submit for review.
      </p>

      <form
        action={actCreateAttestation}
        className="mt-8 grid gap-4 max-w-xl"
      >
        <label className="grid gap-1 text-sm">
          <span className="font-medium">Organization (legal name)</span>
          <input
            name="organization"
            required
            className="rounded border border-ink/20 px-3 py-2 font-mono"
            placeholder="Acme AI, Inc."
          />
        </label>

        <label className="grid gap-1 text-sm">
          <span className="font-medium">Scope type</span>
          <select
            name="scope_type"
            defaultValue="model"
            className="rounded border border-ink/20 px-3 py-2"
          >
            <option value="model">per-model</option>
            <option value="deployment">per-deployment</option>
            <option value="organization">per-organization</option>
          </select>
        </label>

        <label className="grid gap-1 text-sm">
          <span className="font-medium">Scope subject</span>
          <input
            name="scope_subject"
            required
            className="rounded border border-ink/20 px-3 py-2 font-mono"
            placeholder="Acme-RL-v1.0 / claims-bot-prod / Acme AI Inc."
          />
        </label>

        <label className="grid gap-1 text-sm">
          <span className="font-medium">Tier</span>
          <select
            name="tier"
            defaultValue="bronze"
            className="rounded border border-ink/20 px-3 py-2"
          >
            <option value="bronze">Bronze (annual, self-attested)</option>
            <option value="silver">Silver (annual, Vlabs-audited)</option>
            <option value="gold">
              Gold (continuous, third-party-audited)
            </option>
          </select>
        </label>

        <fieldset className="grid gap-2 text-sm">
          <legend className="font-medium">Standards alignment</legend>
          {STANDARDS.map((s) => (
            <label key={s.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                name="standards_requested"
                value={s.id}
              />
              <span>{s.label}</span>
            </label>
          ))}
        </fieldset>

        <button
          type="submit"
          className="mt-4 rounded bg-ink px-4 py-2 text-sm text-white hover:bg-ink/80"
        >
          Create draft
        </button>
      </form>
    </section>
  );
}
