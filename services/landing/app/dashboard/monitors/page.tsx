import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

export const runtime = "edge";

type MonitorSummary = {
  monitor_id: string;
  name: string;
  model_name: string;
  cadence: "daily" | "weekly" | "monthly";
  status: "active" | "paused" | "failed";
  env_subset: string[];
  episodes_per_env: number;
  last_run_at: string | null;
  next_run_at: string;
  created_at: string;
};

async function fetchMonitors(token: string): Promise<MonitorSummary[]> {
  const apiBase = process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiBase}/v1/monitors?limit=100`, {
    headers: { "X-Vlabs-Key": token },
    cache: "no-store",
  });
  if (!r.ok) return [];
  const body = await r.json();
  return body.items ?? [];
}

export default async function MonitorsPage() {
  const { getToken } = await auth();
  const token = await getToken();
  const monitors = token ? await fetchMonitors(token).catch(() => []) : [];

  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Monitors</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Continuous capability monitoring — Verifiable Labs runs the audit on
        your registered model endpoints at the cadence you choose, alerts on
        regressions, and stores the per-run report.
      </p>

      {monitors.length === 0 ? (
        <p className="mt-8 text-sm text-ink-muted">
          No monitors yet. Create one via{" "}
          <code className="font-mono text-xs">POST /v1/monitors</code>.
        </p>
      ) : (
        <div className="mt-8 grid gap-4">
          {monitors.map((m) => (
            <Link
              key={m.monitor_id}
              href={`/dashboard/monitors/${m.monitor_id}`}
              className="card block hover:border-ink"
            >
              <div className="flex items-baseline justify-between">
                <h2 className="text-lg font-semibold">{m.name}</h2>
                <span
                  className={
                    m.status === "active"
                      ? "text-xs uppercase text-green-600"
                      : "text-xs uppercase text-ink-muted"
                  }
                >
                  {m.status}
                </span>
              </div>
              <p className="mt-1 font-mono text-xs text-ink-muted">
                {m.model_name} · {m.cadence} · {m.env_subset.join(", ")} ·{" "}
                {m.episodes_per_env} episodes
              </p>
              <p className="mt-2 text-xs text-ink-muted">
                Next run: {new Date(m.next_run_at).toLocaleString()}
              </p>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}
