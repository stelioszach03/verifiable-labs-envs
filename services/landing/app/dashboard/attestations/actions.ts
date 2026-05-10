"use server";

import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";

import {
  type AttestationArtifactKind,
  type AttestationCycle,
  type AttestationScope,
  type AttestationStandardName,
  type AttestationTier,
  createAttestation,
  initiateRenewal,
  patchAttestation,
  revokeAttestation,
  uploadArtifact,
} from "@/lib/attestations";

async function token(): Promise<string> {
  const { getToken, userId } = await auth();
  if (!userId) {
    throw new Error("Not authenticated");
  }
  // Mirror dashboard/actions.ts: use the "vlabs-api" Clerk template so
  // the bearer token carries the email claim. For now this token is
  // forwarded as X-Vlabs-Key (matches Phase 28 monitors). 31.D will
  // formalise the dashboard-auth bridge.
  const tok = await getToken({ template: "vlabs-api" });
  if (!tok) {
    throw new Error("Clerk did not return a session token");
  }
  return tok;
}

const ALLOWED_TIERS: ReadonlySet<string> = new Set([
  "bronze",
  "silver",
  "gold",
]);
const ALLOWED_SCOPES: ReadonlySet<string> = new Set([
  "model",
  "deployment",
  "organization",
]);
const ALLOWED_STANDARDS: ReadonlySet<string> = new Set([
  "iso_42001",
  "nist_ai_rmf",
  "eu_ai_act",
  "soc2",
]);
const ALLOWED_KINDS: ReadonlySet<string> = new Set([
  "training_doc",
  "audit_report",
  "monitor_record",
  "rm_record",
  "prm_record",
  "change_mgmt",
  "legal_signoff",
  "third_party_audit",
]);
const TIER_TO_CYCLE: Record<string, AttestationCycle> = {
  bronze: "annual",
  silver: "annual",
  gold: "continuous",
};

function readField(form: FormData, name: string): string {
  return String(form.get(name) ?? "").trim();
}

function readStandards(form: FormData): AttestationStandardName[] {
  const raw = form.getAll("standards_requested").map(String);
  const out: AttestationStandardName[] = [];
  for (const s of raw) {
    if (ALLOWED_STANDARDS.has(s)) {
      out.push(s as AttestationStandardName);
    }
  }
  return out;
}

export async function actCreateAttestation(form: FormData): Promise<void> {
  const organization = readField(form, "organization");
  const scope_type = readField(form, "scope_type");
  const scope_subject = readField(form, "scope_subject");
  const tier = readField(form, "tier");
  if (!organization) {
    throw new Error("actCreateAttestation: organization is required");
  }
  if (!scope_subject) {
    throw new Error("actCreateAttestation: scope_subject is required");
  }
  if (!ALLOWED_SCOPES.has(scope_type)) {
    throw new Error(`actCreateAttestation: invalid scope_type=${scope_type}`);
  }
  if (!ALLOWED_TIERS.has(tier)) {
    throw new Error(`actCreateAttestation: invalid tier=${tier}`);
  }
  const cycle: AttestationCycle = TIER_TO_CYCLE[tier];
  const standards_requested = readStandards(form);

  const tok = await token();
  const created = await createAttestation(tok, {
    organization,
    scope_type: scope_type as AttestationScope,
    scope_subject,
    tier: tier as AttestationTier,
    cycle,
    standards_requested,
  });
  revalidatePath("/dashboard/attestations");
  redirect(`/dashboard/attestations/${created.id}`);
}

export async function actSubmitAttestation(form: FormData): Promise<void> {
  const id = readField(form, "id");
  if (!id) {
    throw new Error("actSubmitAttestation: missing id");
  }
  const tok = await token();
  await patchAttestation(tok, id, { action: "submit" });
  revalidatePath(`/dashboard/attestations/${id}`);
  redirect(`/dashboard/attestations/${id}`);
}

export async function actWithdrawAttestation(form: FormData): Promise<void> {
  const id = readField(form, "id");
  if (!id) {
    throw new Error("actWithdrawAttestation: missing id");
  }
  const tok = await token();
  await patchAttestation(tok, id, { action: "withdraw" });
  revalidatePath(`/dashboard/attestations/${id}`);
  redirect(`/dashboard/attestations/${id}`);
}

export async function actUploadArtifact(form: FormData): Promise<void> {
  const id = readField(form, "id");
  const kind = readField(form, "kind");
  const file = form.get("file");
  if (!id) {
    throw new Error("actUploadArtifact: missing id");
  }
  if (!ALLOWED_KINDS.has(kind)) {
    throw new Error(`actUploadArtifact: invalid kind=${kind}`);
  }
  if (!(file instanceof File) || file.size === 0) {
    throw new Error("actUploadArtifact: file is required and non-empty");
  }
  // Edge Runtime: read file as ArrayBuffer, base64 it. The 50 MB cap
  // is enforced server-side per Phase 31 §5 D9-A.
  const buf = new Uint8Array(await file.arrayBuffer());
  let content_b64 = "";
  // Manual base64 because Edge Runtime btoa needs binary strings.
  let s = "";
  for (let i = 0; i < buf.length; i++) s += String.fromCharCode(buf[i]);
  content_b64 = btoa(s);

  const tok = await token();
  await uploadArtifact(tok, id, {
    kind: kind as AttestationArtifactKind,
    filename: file.name,
    content_b64,
    encrypted: readField(form, "encrypted") === "on",
  });
  revalidatePath(`/dashboard/attestations/${id}`);
  revalidatePath(`/dashboard/attestations/${id}/artifacts`);
  redirect(`/dashboard/attestations/${id}`);
}

export async function actInitiateRenewal(form: FormData): Promise<void> {
  const id = readField(form, "id");
  const idempotencyKey = readField(form, "idempotency_key") || crypto.randomUUID();
  if (!id) {
    throw new Error("actInitiateRenewal: missing id");
  }
  const tok = await token();
  await initiateRenewal(tok, id, idempotencyKey);
  revalidatePath(`/dashboard/attestations/${id}`);
  redirect(`/dashboard/attestations/${id}`);
}

export async function actRevokeAttestation(form: FormData): Promise<void> {
  const id = readField(form, "id");
  const reason = readField(form, "reason");
  if (!id) {
    throw new Error("actRevokeAttestation: missing id");
  }
  if (!reason) {
    throw new Error("actRevokeAttestation: reason is required");
  }
  const tok = await token();
  await revokeAttestation(tok, id, reason);
  revalidatePath(`/dashboard/attestations/${id}`);
  redirect(`/dashboard/attestations/${id}`);
}
