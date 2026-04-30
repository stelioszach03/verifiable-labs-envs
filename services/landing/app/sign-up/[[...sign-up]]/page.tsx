import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <section className="container-tight flex justify-center py-20">
      <SignUp appearance={{ elements: { card: "shadow-none" } }} />
    </section>
  );
}
