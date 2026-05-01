import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

export const runtime = "edge";

const API_URL =
  process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";

type DashboardData = {
  counts: {
    users: number;
    api_keys_active: number;
    api_keys_revoked: number;
    calibrations_total: number;
    evaluations_total: number;
    subscriptions_active: number;
  };
  most_recent_calibrations: Array<{
    calibration_id: string;
    api_key_prefix: string;
    n_calibration: number;
    quantile: number;
    created_at: string;
  }>;
  billing_enabled: boolean;
};

async function fetchAdminDashboard():
  Promise<{ status: number; data: DashboardData | null }> {
  const { getToken } = await auth();
  const token = await getToken({ template: "vlabs-api" });
  if (!token) return { status: 401, data: null };

  const res = await fetch(`${API_URL}/v1/admin/dashboard`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!res.ok) return { status: res.status, data: null };
  return { status: 200, data: (await res.json()) as DashboardData };
}

export default async function AdminPage() {
  const { status, data } = await fetchAdminDashboard();

  if (status === 401) {
    return (
      <section className="container-tight py-20 text-center">
        <h1 className="text-2xl font-semibold">Sign in required</h1>
        <p className="mt-3 text-sm text-ink-muted">
          The admin dashboard is gated behind Clerk auth.
        </p>
        <Link href="/sign-in" className="btn-accent mt-6 inline-block">
          Sign in
        </Link>
      </section>
    );
  }
  if (status === 403) {
    return (
      <section className="container-tight py-20 text-center">
        <h1 className="text-2xl font-semibold">Not authorized</h1>
        <p className="mt-3 text-sm text-ink-muted">
          Your Clerk user ID is not in <code>VLABS_ADMIN_CLERK_IDS</code>.
          Ask the owner to add it.
        </p>
      </section>
    );
  }
  if (!data) {
    return (
      <section className="container-tight py-20 text-center">
        <h1 className="text-2xl font-semibold">Admin dashboard unavailable</h1>
        <p className="mt-3 text-sm text-ink-muted">
          API responded with status {status}. Check `flyctl logs --app vlabs-api`.
        </p>
      </section>
    );
  }

  const c = data.counts;

  return (
    <section className="container-tight py-12">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Live counts from the production API. Read-only.
          </p>
        </div>
        <span
          className={`rounded-full px-3 py-1 text-xs ${
            data.billing_enabled
              ? "bg-green-100 text-green-900"
              : "bg-amber-100 text-amber-900"
          }`}
        >
          billing: {data.billing_enabled ? "live" : "deferred"}
        </span>
      </header>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        <Stat label="Users" value={c.users} />
        <Stat label="API keys active" value={c.api_keys_active} />
        <Stat label="API keys revoked" value={c.api_keys_revoked} />
        <Stat label="Calibrations total" value={c.calibrations_total} />
        <Stat label="Evaluations total" value={c.evaluations_total} />
        <Stat label="Subscriptions active" value={c.subscriptions_active} />
      </div>

      <h2 className="mt-12 text-lg font-semibold">Recent calibrations</h2>
      {data.most_recent_calibrations.length === 0 ? (
        <p className="mt-4 text-sm text-ink-muted">
          No calibrations yet — the first customer call will populate this.
        </p>
      ) : (
        <table className="mt-4 w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-ink-muted">
            <tr>
              <th className="pb-3">Calibration</th>
              <th className="pb-3">Key prefix</th>
              <th className="pb-3">n_calibration</th>
              <th className="pb-3">quantile</th>
              <th className="pb-3">created</th>
            </tr>
          </thead>
          <tbody>
            {data.most_recent_calibrations.map((row) => (
              <tr
                key={row.calibration_id}
                className="border-t border-ink/10"
              >
                <td className="py-3 font-mono text-xs">
                  {row.calibration_id.slice(0, 16)}…
                </td>
                <td className="py-3 font-mono text-xs">{row.api_key_prefix}</td>
                <td className="py-3">{row.n_calibration.toLocaleString()}</td>
                <td className="py-3">{row.quantile.toFixed(4)}</td>
                <td className="py-3 text-ink-muted">
                  {new Date(row.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value.toLocaleString()}</p>
    </div>
  );
}
