import { auth } from "@clerk/nextjs/server";
import { getUsage } from "@/lib/api";

export default async function UsagePage() {
  const { getToken } = await auth();
  const token = await getToken();
  const usage = token ? await getUsage(token).catch(() => null) : null;

  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Usage</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Counters reset on the first day of each calendar month.
      </p>

      {usage === null ? (
        <p className="mt-8 text-sm text-ink-muted">
          Could not load usage. Make sure the API is running and you are signed in.
        </p>
      ) : (
        <>
          <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Traces"
              value={usage.usage.traces.toLocaleString()}
              caption={`of ${usage.quota.traces_per_month.toLocaleString()} quota`}
            />
            <Stat
              label="Calibrations"
              value={usage.usage.calibrations.toLocaleString()}
            />
            <Stat
              label="Predictions"
              value={usage.usage.predictions.toLocaleString()}
            />
            <Stat
              label="Evaluations"
              value={usage.usage.evaluations.toLocaleString()}
            />
          </div>
          <div className="mt-10 card">
            <p className="text-xs uppercase tracking-wide text-ink-muted">
              Period
            </p>
            <p className="mt-1 font-mono text-sm">
              {usage.current_period.start} → {usage.current_period.end}
            </p>
            <p className="mt-4 text-xs uppercase tracking-wide text-ink-muted">
              Tier
            </p>
            <p className="mt-1 capitalize">
              {usage.tier} · {usage.quota.rpm.toLocaleString()} req / min
            </p>
            <p className="mt-4 text-xs uppercase tracking-wide text-ink-muted">
              Remaining traces
            </p>
            <p className="mt-1">
              {usage.remaining.traces.toLocaleString()}
            </p>
          </div>
        </>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption?: string;
}) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
      {caption ? <p className="mt-1 text-xs text-ink-muted">{caption}</p> : null}
    </div>
  );
}
