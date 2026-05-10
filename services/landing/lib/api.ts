/**
 * Typed wrapper around vlabs-api. All calls are made from server actions
 * with the user's Clerk session token in the Authorization header. The
 * dashboard never has a Clerk JWT in client-side code paths — that's
 * the whole point of running these as server actions.
 */

const BASE_URL =
  process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";

export type APIKeyInfo = {
  id: string;
  prefix: string;
  name: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
};

export type APIKeyCreated = APIKeyInfo & { plaintext_key: string };

export type UsageResponse = {
  tier: "free" | "pro" | "team";
  quota: { traces_per_month: number; rpm: number };
  current_period: { start: string; end: string };
  usage: { traces: number; calibrations: number; evaluations: number; predictions: number };
  remaining: { traces: number };
};

type FetchOptions = {
  token: string;
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
};

async function call<T>(path: string, opts: FetchOptions): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: opts.method ?? "GET",
    headers: {
      Authorization: `Bearer ${opts.token}`,
      "Content-Type": "application/json",
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail ?? body.title ?? JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`vlabs-api ${path} -> ${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export async function listApiKeys(token: string): Promise<APIKeyInfo[]> {
  const out = await call<{ items: APIKeyInfo[] }>("/v1/keys", { token });
  return out.items;
}

export async function createApiKey(
  token: string,
  name: string,
): Promise<APIKeyCreated> {
  return call<APIKeyCreated>("/v1/keys", {
    token,
    method: "POST",
    body: { name },
  });
}

export async function revokeApiKey(
  token: string,
  id: string,
): Promise<APIKeyInfo> {
  return call<APIKeyInfo>(`/v1/keys/${id}`, { token, method: "DELETE" });
}

export async function getUsage(token: string): Promise<UsageResponse> {
  return call<UsageResponse>("/v1/usage", { token });
}

export async function startCheckout(
  token: string,
  tier: "pro" | "team",
): Promise<{ url: string; tier: string }> {
  return call<{ url: string; tier: string }>("/v1/billing/checkout", {
    token,
    method: "POST",
    body: { tier },
  });
}

export async function startBillingPortal(
  token: string,
): Promise<{ url: string }> {
  return call<{ url: string }>("/v1/billing/portal", {
    token,
    method: "POST",
  });
}
