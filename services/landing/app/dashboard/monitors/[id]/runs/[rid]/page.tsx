import { auth } from "@clerk/nextjs/server";
import Link from "next/link";

export const runtime = "edge";

type RunDetail = {
  monitor_run_id: string;
  monitor_id: string;
  scheduled_at: string;
  started_at: string | null;
  finished_at: string | null;
  status: "queued" | "running" | "success" | "failed";
  summary_stats: Record<string, unknown> | null;
  regression_verdict: "ok" | "warning" | "regressed" | null;
  verdict_payload: {
    conformal?: { current?: number; baseline?: number; delta_to_target?: number };
    bootstrap?: {
      mean_delta?: number; ci_low?: number; ci_high?: number; p_value?: number;
    };
  } | null;
  pdf_url: string | null;
  pdf_sha256: string | null;
  cost_usd_estimate: number | null;
  error: string | null;
  trigger: "scheduled" | "manual";
};

async function fetchRun(
  token: string, monitorId: string, runId: string,
): Promise<RunDetail | null> {
  const apiBase = process.env.NEXT_PUBLIC_VLABS_API_URL ?? "http://localhost:8000";
  const r = await fetch(
    `${apiBase}/v1/monitors/${monitorId}/runs/${runId}`,
    { headers: { "X-Vlabs-Key": token }, cache: "no-store" },
  );
  if (!r.ok) return null;
  return (await r.json()) as RunDetail;
}

export default async function MonitorRunPage({
  params,
}: {
  params: Promise<{ id: string; rid: string }>;
}) {
  const { id, rid } = await params;
  const { getToken } = await auth();
  const token = await getToken();
  const run = token ? await fetchRun(token, id, rid) : null;

  if (run === null) {
    return (
      <section>
        <Link
          href={`/dashboard/monitors/${id}`}
          className="text-xs text-ink-muted hover:text-ink"
        >
          ← monitor
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">
          Run not found
        </h1>
      </section>
    );
  }

  const verdict = run.regression_verdict ?? run.status;

  return (
    <section>
      <Link
        href={`/dashboard/monitors/${id}`}
        className="text-xs text-ink-muted hover:text-ink"
      >
        ← monitor
      </Link>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">
        {run.monitor_run_id}
      </h1>
      <p className="mt-1 text-xs text-ink-muted">
        {run.trigger} · {new Date(run.scheduled_at).toLocaleString()} ·{" "}
        <span
          className={
            verdict === "regressed"
              ? "uppercase text-red-600"
              : verdict === "warning"
                ? "uppercase text-amber-600"
                : "uppercase text-green-600"
          }
        >
          {verdict}
        </span>
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {run.verdict_payload?.conformal ? (
          <div className="card">
            <p className="text-xs uppercase tracking-wide text-ink-muted">
              Conformal coverage
            </p>
            <p className="mt-1 text-sm">
              current = {run.verdict_payload.conformal.current ?? "—"}
            </p>
            <p className="text-sm">
              baseline = {run.verdict_payload.conformal.baseline ?? "—"}
            </p>
            <p className="text-sm">
              Δ vs target = {run.verdict_payload.conformal.delta_to_target ?? "—"}
            </p>
          </div>
        ) : null}
        {run.verdict_payload?.bootstrap ? (
          <div className="card">
            <p className="text-xs uppercase tracking-wide text-ink-muted">
              Bootstrap reward Δ
            </p>
            <p className="mt-1 text-sm">
              mean_delta = {run.verdict_payload.bootstrap.mean_delta ?? "—"}
            </p>
            <p className="text-sm">
              95% CI = ({run.verdict_payload.bootstrap.ci_low ?? "—"},{" "}
              {run.verdict_payload.bootstrap.ci_high ?? "—"})
            </p>
            <p className="text-sm">
              p_value = {run.verdict_payload.bootstrap.p_value ?? "—"}
            </p>
          </div>
        ) : null}
      </div>

      {run.pdf_url ? (
        <p className="mt-6 text-sm">
          <a
            href={run.pdf_url}
            target="_blank"
            rel="noreferrer"
            className="underline"
          >
            Download PDF report
          </a>
        </p>
      ) : null}

      {run.error ? (
        <pre className="mt-6 whitespace-pre-wrap rounded border border-red-300 p-3 text-xs text-red-700">
          {run.error}
        </pre>
      ) : null}
    </section>
  );
}
