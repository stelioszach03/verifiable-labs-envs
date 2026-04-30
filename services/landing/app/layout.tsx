import type { Metadata } from "next";
import { ClerkProvider, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Verifiable Labs",
  description:
    "Conformal coverage guarantees for any reward function. Five lines of Python.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>
          <header className="border-b border-ink/10 bg-paper">
            <div className="container-tight flex h-16 items-center justify-between">
              <Link href="/" className="text-lg font-semibold tracking-tight">
                Verifiable Labs
              </Link>
              <nav className="flex items-center gap-6 text-sm">
                <Link href="/pricing" className="text-ink-muted hover:text-ink">
                  Pricing
                </Link>
                <a
                  href="https://github.com/stelioszach03/verifiable-labs-envs"
                  className="text-ink-muted hover:text-ink"
                  rel="noreferrer"
                  target="_blank"
                >
                  GitHub
                </a>
                <SignedOut>
                  <Link href="/sign-in" className="text-ink-muted hover:text-ink">
                    Sign in
                  </Link>
                  <Link href="/sign-up" className="btn-accent">
                    Get started
                  </Link>
                </SignedOut>
                <SignedIn>
                  <Link href="/dashboard" className="text-ink-muted hover:text-ink">
                    Dashboard
                  </Link>
                  <UserButton afterSignOutUrl="/" />
                </SignedIn>
              </nav>
            </div>
          </header>
          <main>{children}</main>
          <footer className="mt-24 border-t border-ink/10 py-10 text-sm text-ink-muted">
            <div className="container-tight flex flex-wrap items-center justify-between gap-4">
              <span>© Verifiable Labs · Apache-2.0 SDK</span>
              <span>
                Built by{" "}
                <a
                  href="https://github.com/stelioszach03"
                  className="hover:text-ink"
                  rel="noreferrer"
                  target="_blank"
                >
                  @stelioszach03
                </a>
              </span>
            </div>
          </footer>
        </body>
      </html>
    </ClerkProvider>
  );
}
