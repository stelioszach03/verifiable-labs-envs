import { auth } from "@clerk/nextjs/server";
import { listApiKeys } from "@/lib/api";
import { actCreateKey, actRevokeKey } from "../actions";

export const runtime = "edge";

type SearchParams = { new?: string };

export default async function APIKeysPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const { getToken } = await auth();
  const token = await getToken();
  const keys = token ? await listApiKeys(token).catch(() => []) : [];

  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">API keys</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Use the plaintext key as the <code className="rounded bg-ink/5 px-1 font-mono text-xs">X-Vlabs-Key</code>{" "}
        header. The plaintext is shown <strong>once</strong> on creation —
        copy it immediately and store it in your secret manager.
      </p>

      {params?.new ? (
        <div className="mt-6 rounded-xl border border-accent/40 bg-accent/5 p-4">
          <p className="text-xs uppercase tracking-wide text-accent">
            New API key — copy now, will not be shown again
          </p>
          <code className="mt-2 block break-all rounded-lg bg-white px-3 py-2 font-mono text-sm">
            {params.new}
          </code>
        </div>
      ) : null}

      <form action={actCreateKey} className="mt-8 flex gap-2">
        <input
          type="text"
          name="name"
          placeholder="Key name (e.g. production-trainer)"
          className="flex-1 rounded-lg border border-ink/15 bg-white px-3 py-2 text-sm focus:border-accent focus:outline-none"
        />
        <button type="submit" className="btn-accent">
          Create key
        </button>
      </form>

      <div className="mt-10">
        {keys.length === 0 ? (
          <p className="text-sm text-ink-muted">No keys yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-ink-muted">
              <tr>
                <th className="pb-3">Name</th>
                <th className="pb-3">Prefix</th>
                <th className="pb-3">Created</th>
                <th className="pb-3">Last used</th>
                <th className="pb-3">Status</th>
                <th className="pb-3"></th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k.id} className="border-t border-ink/10">
                  <td className="py-3">{k.name}</td>
                  <td className="py-3 font-mono text-xs">{k.prefix}…</td>
                  <td className="py-3">
                    {new Date(k.created_at).toLocaleDateString()}
                  </td>
                  <td className="py-3">
                    {k.last_used_at
                      ? new Date(k.last_used_at).toLocaleString()
                      : "never"}
                  </td>
                  <td className="py-3">
                    {k.revoked_at ? (
                      <span className="text-ink-muted">revoked</span>
                    ) : (
                      <span className="text-green-700">active</span>
                    )}
                  </td>
                  <td className="py-3 text-right">
                    {!k.revoked_at ? (
                      <form action={actRevokeKey}>
                        <input type="hidden" name="id" value={k.id} />
                        <button
                          type="submit"
                          className="text-xs text-ink-muted hover:text-red-600"
                        >
                          Revoke
                        </button>
                      </form>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
