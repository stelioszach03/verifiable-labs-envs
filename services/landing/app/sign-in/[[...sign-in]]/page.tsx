import { SignIn } from "@clerk/nextjs";

export const runtime = "edge";

export default function SignInPage() {
  return (
    <section className="container-tight flex justify-center py-20">
      <SignIn appearance={{ elements: { card: "shadow-none" } }} />
    </section>
  );
}
