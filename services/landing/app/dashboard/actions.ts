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

export async function actRevokeKey(id: string): Promise<void> {
  const tok = await token();
  await revokeApiKey(tok, id);
  revalidatePath("/dashboard/api-keys");
}

export async function actUpgradeTo(tier: "pro" | "team"): Promise<void> {
  const tok = await token();
  const { url } = await startCheckout(tok, tier);
  redirect(url);
}

export async function actOpenPortal(): Promise<void> {
  const tok = await token();
  const { url } = await startBillingPortal(tok);
  redirect(url);
}
