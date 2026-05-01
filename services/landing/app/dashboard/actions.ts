"use server";

import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";

import {
  createApiKey,
  revokeApiKey,
  startBillingPortal,
  startCheckout,
} from "@/lib/api";

async function token(): Promise<string> {
  const { getToken, userId } = await auth();
  if (!userId) {
    throw new Error("Not authenticated");
  }
  // Use the "vlabs-api" JWT template so the token carries the user's
  // email claim — vlabs_api/clerk_auth.py reads it for JIT user creation.
  const tok = await getToken({ template: "vlabs-api" });
  if (!tok) {
    throw new Error("Clerk did not return a session token");
  }
  return tok;
}

export async function actCreateKey(formData: FormData): Promise<void> {
  const name = String(formData.get("name") ?? "").trim() || "default";
  const tok = await token();
  const created = await createApiKey(tok, name);
  // Persist the plaintext into the URL once so the page can show it.
  revalidatePath("/dashboard/api-keys");
  redirect(`/dashboard/api-keys?new=${encodeURIComponent(created.plaintext_key)}`);
}

// All server actions take FormData (the standard Next.js 15 / Cloudflare
// Workers signature). Inline server-action closures inside loops break
// edge-runtime serialization in next-on-pages — the bound parameters
// cannot be encrypted/embedded reliably. Pages must use a hidden <input
// name="..." /> instead of a closure capture.

export async function actRevokeKey(formData: FormData): Promise<void> {
  const id = String(formData.get("id") ?? "");
  if (!id) {
    throw new Error("actRevokeKey: missing 'id' in form payload");
  }
  const tok = await token();
  await revokeApiKey(tok, id);
  revalidatePath("/dashboard/api-keys");
}

export async function actUpgradeTo(formData: FormData): Promise<void> {
  const tier = String(formData.get("tier") ?? "") as "pro" | "team";
  if (tier !== "pro" && tier !== "team") {
    throw new Error(`actUpgradeTo: invalid tier ${JSON.stringify(tier)}`);
  }
  const tok = await token();
  const { url } = await startCheckout(tok, tier);
  redirect(url);
}

export async function actOpenPortal(_formData?: FormData): Promise<void> {
  const tok = await token();
  const { url } = await startBillingPortal(tok);
  redirect(url);
}
