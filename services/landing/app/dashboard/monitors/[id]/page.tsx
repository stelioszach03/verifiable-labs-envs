import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

export const runtime = "edge";

type AlertChannel = {
  type: "email" | "slack" | "webhook";
  address?: string;
  webhook_url_fingerprint?: string;
};

type MonitorDetail = {
  monitor_id: string;
  name: string;
  model_endpoint: string;
  model_name: string;
  auth_token_fingerprint: string;
  cadence: "daily" | "weekly" | "monthly";
  env_subset: string[];
  episodes_per_env: number;
  alert_channels: AlertChannel[];
  status: "active" | "paused" | "failed";
  retention_days: number;
  baseline_run_id: string | null;
  created_at: string;
  updated_at: string;
  last_run_at: string | null;
  next_run_at: string;
  projected_monthly_episodes: number;
};

type RunSummary = {
  monitor_run_id: string;
  scheduled_at: string;
  status: string;
  trigger: string;
  regression_verdict: string | null;
  cost_usd_estimate: number | null;
};

async function fetchMonitor(
  token: string, monitorId: string,
): Promise<MonitorDetail | null> {
  const apiBase = process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiBase}/v1/monitors/${monitorId}`, {
    headers: { "X-Vlabs-Key": token },
    cache: "no-store",
  });
  if (!r.ok) return null;
  return (await r.json()) as MonitorDetail;
}

async function fetchRuns(
  token: string, monitorId: string,
): Promise<RunSummary[]> {
  const apiBase = process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";
  const r = await fetch(`${apiBase}/v1/monitors/${monitorId}/runs?limit=20`, {
    headers: { "X-Vlabs-Key": token },
    cache: "no-store",
  });
  if (!r.ok) return [];
  const body = await r.json();
  return body.items ?? [];
}

export default async function MonitorDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { getToken } = await auth();
  const token = await getToken();
  const monitor = token ? await fetchMonitor(token, id) : null;
  const runs = token ? await fetchRuns(token, id) : [];

  if (monitor === null) {
    return (
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Monitor</h1>
        <p className="mt-4 text-sm text-ink-muted">
          Monitor not found or you are not authenticated.
        </p>
      </section>
    );
  }

  return (
    <section>
      <Link
        href="/dashboard/monitors"
        className="text-xs text-ink-muted hover:text-ink"
      >
        ← all monitors
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">
        {monitor.name}
      </h1>
      <p className="mt-1 font-mono text-xs text-ink-muted">{monitor.monitor_id}</p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">Model</p>
          <p className="mt-1 font-mono text-sm">{monitor.model_name}</p>
          <p className="mt-1 font-mono text-xs text-ink-muted">
            {monitor.model_endpoint}
          </p>
          <p className="mt-2 text-xs text-ink-muted">
            Token fingerprint: {monitor.auth_token_fingerprint}
          </p>
        </div>
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">Cadence</p>
          <p className="mt-1 text-sm">
            {monitor.cadence} · {monitor.env_subset.join(", ")} ·{" "}
            {monitor.episodes_per_env} episodes
          </p>
          <p className="mt-2 text-xs text-ink-muted">
            Projected ≈ {monitor.projected_monthly_episodes} episodes / month
          </p>
        </div>
      </div>

      <h2 className="mt-10 text-lg font-semibold tracking-tight">Recent runs</h2>
      {runs.length === 0 ? (
        <p className="mt-2 text-sm text-ink-muted">
          No runs yet — wait for the next scheduled fire or trigger one with{" "}
          <code className="font-mono text-xs">
            POST /v1/monitors/{monitor.monitor_id}/run
          </code>
          .
        </p>
      ) : (
        <ul className="mt-4 grid gap-2">
          {runs.map((r) => (
            <li key={r.monitor_run_id}>
              <Link
                href={`/dashboard/monitors/${id}/runs/${r.monitor_run_id}`}
                className="card block hover:border-ink"
              >
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-xs">{r.monitor_run_id}</span>
                  <span
                    className={
                      r.regression_verdict === "regressed"
                        ? "text-xs uppercase text-red-600"
                        : r.regression_verdict === "warning"
                          ? "text-xs uppercase text-amber-600"
                          : "text-xs uppercase text-green-600"
                    }
                  >
                    {r.regression_verdict ?? r.status}
                  </span>
                </div>
                <p className="mt-1 text-xs text-ink-muted">
                  {new Date(r.scheduled_at).toLocaleString()} · {r.trigger}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
