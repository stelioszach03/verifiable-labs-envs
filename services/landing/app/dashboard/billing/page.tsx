import { auth } from "@clerk/nextjs/server";
import { getUsage } from "@/lib/api";
import { actOpenPortal, actUpgradeTo } from "../actions";

export const runtime = "edge";

export default async function BillingPage() {
  const { getToken } = await auth();
  const token = await getToken();
  const usage = token ? await getUsage(token).catch(() => null) : null;
  const tier = usage?.tier ?? "free";

  return (
    <section>
      <h1 className="text-2xl font-semibold tracking-tight">Billing</h1>
      <p className="mt-1 text-sm text-ink-muted">
        Stripe is in <strong>test mode</strong> — checkout flows complete
        but no real money moves until the Verifiable Labs Inc. (Delaware
        C-corp) registration finishes.
      </p>

      <div className="mt-8 grid gap-6 md:grid-cols-2">
        <div className="card">
          <p className="text-xs uppercase tracking-wide text-ink-muted">
            Current plan
          </p>
          <p className="mt-1 text-2xl font-semibold capitalize">{tier}</p>
          {tier !== "free" ? (
            <form
              action={async () => {
                "use server";
                await actOpenPortal();
              }}
              className="mt-4"
            >
              <button type="submit" className="btn-ghost">
                Open Stripe billing portal
              </button>
            </form>
          ) : (
            <p className="mt-2 text-sm text-ink-muted">
              You're on the free tier. Upgrade below.
            </p>
          )}
        </div>

        <UpgradeCard
          tier="pro"
          price="$99 / mo"
          desc="1M traces / month + metered overage at $1 / 10K."
          disabled={tier === "pro"}
        />

        <UpgradeCard
          tier="team"
          price="$499 / mo"
          desc="10M traces / month + metered overage at $0.40 / 10K."
          disabled={tier === "team"}
        />
      </div>
    </section>
  );
}

function UpgradeCard({
  tier,
  price,
  desc,
  disabled,
}: {
  tier: "pro" | "team";
  price: string;
  desc: string;
  disabled: boolean;
}) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-ink-muted">
        {tier === "pro" ? "Pro" : "Team"}
      </p>
      <p className="mt-1 text-2xl font-semibold">{price}</p>
      <p className="mt-2 text-sm text-ink-muted">{desc}</p>
      <form
        action={async () => {
          "use server";
          await actUpgradeTo(tier);
        }}
        className="mt-4"
      >
        <button
          type="submit"
          className={tier === "pro" ? "btn-accent" : "btn-ghost"}
          disabled={disabled}
        >
          {disabled ? "Current plan" : `Upgrade to ${tier === "pro" ? "Pro" : "Team"}`}
        </button>
      </form>
    </div>
  );
}
