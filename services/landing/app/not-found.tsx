import Link from "next/link";

export const runtime = "edge";

export default function NotFound() {
  return (
    <section className="container-tight py-32 text-center">
      <p className="text-sm uppercase tracking-wide text-ink-muted">404</p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight">
        Page not found
      </h1>
      <p className="mx-auto mt-4 max-w-md text-ink-muted">
        Either the URL is wrong, or the page hasn't been built yet. The API
        is at <code className="rounded bg-ink/5 px-1 font-mono text-xs">api.verifiable-labs.com</code>.
      </p>
      <Link href="/" className="btn-accent mt-8 inline-block">
        Back to landing
      </Link>
    </section>
  );
}
