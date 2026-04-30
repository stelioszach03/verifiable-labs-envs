import Link from "next/link";

const TABS = [
  { href: "/dashboard", label: "Overview" },
  { href: "/dashboard/api-keys", label: "API keys" },
  { href: "/dashboard/usage", label: "Usage" },
  { href: "/dashboard/billing", label: "Billing" },
];

export default function DashboardLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="container-tight py-12">
      <nav className="mb-10 flex gap-1 border-b border-ink/10">
        {TABS.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className="px-4 py-3 text-sm text-ink-muted hover:text-ink"
          >
            {t.label}
          </Link>
        ))}
      </nav>
      {children}
    </div>
  );
}
