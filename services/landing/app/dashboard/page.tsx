import Link from "next/link";
import { auth } from "@clerk/nextjs/server";

import { getUsage, listApiKeys } from "@/lib/api";

async function fetchOverview() {
  const { getToken } = await auth();
  const token = await getToken();
  if (!token) return null;
  const [usage, keys] = await Promise.all([
    getUsage(token).catch(() => null),
    listApiKeys(token).catch(() => []),
  ]);
  return { usage, keys };
}

export default async function DashboardOverview() {
  const data = await fetchOverview();

  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Quick view of your tier, usage, and API keys.
      </p>

      <div className="mt-8 grid gap-6 md:grid-cols-3">
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">
            Tier
          </p>
          <p className="mt-1 text-2xl font-semibold capitalize">
            {data?.usage?.tier ?? "—"}
          </p>
          <Link
            href="/dashboard/billing"
            className="mt-4 inline-block text-sm text-accent hover:text-accent-hover"
          >
            Manage plan →
          </Link>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">
            Traces this month
          </p>
          <p className="mt-1 text-2xl font-semibold">
            {data?.usage ? data.usage.usage.traces.toLocaleString() : "—"}
          </p>
          <p className="mt-1 text-xs text-ink-muted">
            of{" "}
            {data?.usage
              ? data.usage.quota.traces_per_month.toLocaleString()
              : "—"}{" "}
            quota
          </p>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">
            Active API keys
          </p>
          <p className="mt-1 text-2xl font-semibold">
            {data?.keys.filter((k) => !k.revoked_at).length ?? 0}
          </p>
          <Link
            href="/dashboard/api-keys"
            className="mt-4 inline-block text-sm text-accent hover:text-accent-hover"
          >
            Manage keys →
          </Link>
        </div>
      </div>
    </section>
  );
}
