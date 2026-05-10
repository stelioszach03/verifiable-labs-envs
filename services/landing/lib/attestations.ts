/**
 * Typed wrapper around the V-Certified attestation owner endpoints
 * (Phase 31.B). All seven endpoints are auth'd via X-Vlabs-Key on the
 * data plane. Server actions in app/dashboard/attestations/actions.ts
 * read a Clerk template token and pass it through; cf. monitors/.
 */

const BASE_URL =
  process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";

export type AttestationScope = "model" | "deployment" | "organization";
export type AttestationTier = "bronze" | "silver" | "gold";
export type AttestationCycle = "annual" | "continuous";
export type AttestationStatusValue =
  | "draft"
  | "submitted"
  | "under_review"
  | "approved"
  | "revoked"
  | "expired"
  | "withdrawn";
export type AttestationStandardName =
  | "iso_42001"
  | "nist_ai_rmf"
  | "eu_ai_act"
  | "soc2";
export type AttestationArtifactKind =
  | "training_doc"
  | "audit_report"
  | "monitor_record"
  | "rm_record"
  | "prm_record"
  | "change_mgmt"
  | "legal_signoff"
  | "third_party_audit";

export type AttestationStandardsAlignment = {
  standards: AttestationStandardName[];
  crosswalk_version: string | null;
  framework_versions: Record<string, string>;
};

export type AttestationSummary = {
  id: string;
  public_id: string;
  organization: string;
  scope_type: AttestationScope;
  scope_subject: string;
  tier: AttestationTier;
  status: AttestationStatusValue;
  cycle: AttestationCycle;
  issued_at: string | null;
  expires_at: string | null;
  created_at: string;
};

export type AttestationInfo = AttestationSummary & {
  revoked_at: string | null;
  revocation_reason: string | null;
  cert_serial: string | null;
  standards_alignment: AttestationStandardsAlignment;
  artifact_count: number;
};

export type AttestationList = {
  items: AttestationSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type AttestationArtifactInfo = {
  id: string;
  attestation_id: string;
  kind: AttestationArtifactKind;
  storage_uri: string;
  sha256_hash: string;
  encrypted: boolean;
  size_bytes: number;
  submitted_at: string;
};

export type AttestationRenewalInfo = {
  id: string;
  attestation_id: string;
  cycle_number: number;
  initiated_at: string;
  completed_at: string | null;
  new_cert_serial: string | null;
};

type FetchOptions = {
  token: string;
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
};

async function call<T>(path: string, opts: FetchOptions): Promise<T> {
  const r = await fetch(`${BASE_URL}${path}`, {
    method: opts.method ?? "GET",
    headers: {
      "X-Vlabs-Key": opts.token,
      "Content-Type": "application/json",
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    cache: "no-store",
  });
  if (!r.ok) {
    let detail = "";
    try {
      const body = await r.json();
      detail = body.detail ?? body.title ?? JSON.stringify(body);
    } catch {
      detail = await r.text();
    }
    throw new Error(`vlabs-api ${path} -> ${r.status}: ${detail}`);
  }
  return (await r.json()) as T;
}

export async function listAttestations(
  token: string,
): Promise<AttestationList> {
  return call<AttestationList>("/v1/attestations?limit=100", { token });
}

export async function getAttestation(
  token: string,
  id: string,
): Promise<AttestationInfo> {
  return call<AttestationInfo>(`/v1/attestations/${id}`, { token });
}

export type CreateAttestationPayload = {
  organization: string;
  scope_type: AttestationScope;
  scope_subject: string;
  tier: AttestationTier;
  cycle: AttestationCycle;
  standards_requested: AttestationStandardName[];
};

export async function createAttestation(
  token: string,
  payload: CreateAttestationPayload,
): Promise<AttestationInfo> {
  return call<AttestationInfo>("/v1/attestations", {
    token,
    method: "POST",
    body: payload,
  });
}

export type PatchAttestationPayload = {
  action?: "submit" | "withdraw";
  organization?: string;
  scope_subject?: string;
  standards_requested?: AttestationStandardName[];
};

export async function patchAttestation(
  token: string,
  id: string,
  payload: PatchAttestationPayload,
): Promise<AttestationInfo> {
  return call<AttestationInfo>(`/v1/attestations/${id}`, {
    token,
    method: "PATCH",
    body: payload,
  });
}

export type UploadArtifactPayload = {
  kind: AttestationArtifactKind;
  filename: string;
  content_b64: string;
  encrypted?: boolean;
};

export async function uploadArtifact(
  token: string,
  attestationId: string,
  payload: UploadArtifactPayload,
): Promise<AttestationArtifactInfo> {
  return call<AttestationArtifactInfo>(
    `/v1/attestations/${attestationId}/artifacts`,
    {
      token,
      method: "POST",
      body: payload,
    },
  );
}

export async function initiateRenewal(
  token: string,
  attestationId: string,
  idempotencyKey: string,
): Promise<AttestationRenewalInfo> {
  return call<AttestationRenewalInfo>(
    `/v1/attestations/${attestationId}/renew`,
    {
      token,
      method: "POST",
      body: { idempotency_key: idempotencyKey },
    },
  );
}

export async function revokeAttestation(
  token: string,
  attestationId: string,
  reason: string,
): Promise<AttestationInfo> {
  return call<AttestationInfo>(`/v1/attestations/${attestationId}`, {
    token,
    method: "DELETE",
    body: { revocation_reason: reason },
  });
}

/** Bytes -> base64 (Edge Runtime supports btoa). */
export function bytesToBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

/** Status badge classnames per attestation lifecycle state. */
export function statusColorClass(status: AttestationStatusValue): string {
  switch (status) {
    case "approved":
      return "text-green-600";
    case "revoked":
    case "expired":
      return "text-red-600";
    case "submitted":
    case "under_review":
      return "text-amber-600";
    case "draft":
    case "withdrawn":
    default:
      return "text-ink-muted";
  }
}
