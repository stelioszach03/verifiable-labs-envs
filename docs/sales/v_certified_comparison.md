# V-Certified vs alternatives

A quick reference comparing the V-Certified programme against existing
AI-system attestation paths customers might evaluate.

## Comparison matrix

| Programme                | Issuer            | Cycle         | Cert format       | Public verify | Cost          |
|--------------------------|-------------------|---------------|-------------------|---------------|---------------|
| V-Certified Bronze       | Verifiable Labs   | Annual        | X.509 + registry  | Yes           | $2,500/yr     |
| V-Certified Silver       | Verifiable Labs   | Annual        | X.509 + registry  | Yes           | $15,000/yr    |
| V-Certified Gold         | 3rd-party + Vlabs | Continuous    | X.509 + registry  | Yes           | $45,000+/yr   |
| ISO/IEC 42001            | Accredited body   | Annual        | Paper certificate | No            | $30k-$80k     |
| SOC 2 Type 2             | AICPA CPA         | Annual        | Letter report     | No (NDA)      | $40k-$200k    |
| EU AI Act Article 43     | Notified body     | Pre-deploy    | Conformity assessment | Mixed     | $50k-$500k    |
| NIST AI RMF              | Self              | Voluntary     | Self-attestation  | No            | $0            |

## When to choose V-Certified

- **You need a public verification artifact**. V-Certified's public
  registry + cryptographic certificate is the only programme on the
  list that publishes machine-verifiable proof at issuance.
- **You want continuous monitoring evidence baked in**. V-Certified
  Silver + Gold integrate with Phase 28 monitors so the certification
  reflects the live state of the model, not a point-in-time snapshot.
- **You've done the work and want a low-friction third-party
  attestation**. Customers maintaining ISO 42001 / SOC 2 / EU AI Act
  conformance typically have most evidence already; V-Certified
  reuses it through the standards crosswalks.
- **You can't afford a full ISO 42001 / SOC 2 cycle yet**. V-Certified
  Bronze ($2,500) is 1/12th the cost of an entry-level SOC 2 Type 1.

## When to choose an alternative

- **Customers explicitly require ISO 42001 certification**. V-Certified
  is not an accredited certification body. Use an ISO-accredited body.
- **EU AI Act high-risk system pre-market**. Article 43 requires a
  notified-body assessment. V-Certified is INPUT, not the assessment
  itself.
- **Your buyers will only accept a SOC 2 Type 2 letter**. Same: use an
  AICPA CPA. V-Certified can supplement but not replace.

## Differentiators

### Public verification

V-Certified is the only programme on the list that publishes
verifiable cryptographic proof. Verifiers fetch the cert PEM, chain it
to the public CA, and check the public CRL — all in seconds, all
without signing an NDA.

### Continuous monitoring evidence

Silver + Gold tiers tie the attestation to live monitor records from
the Phase 28 monitor system. If the monitor flags a regression, the
review board can fast-track a request-more decision back to draft
status. Other programmes are point-in-time + revoke-on-incident only.

### Standards crosswalks

The four-framework crosswalk model means customers maintaining one
external standard (say SOC 2) can map their existing evidence directly
into V-Certified evidence kinds without re-doing the audit. The
framework_versions field on every attestation pins the upstream
framework version current at issuance for verifier-side
reproducibility.

### Tiered rigor

Bronze gives early-stage startups a path to a public attestation
without the cost of a full audit cycle. Gold gives high-stakes
deployers a continuous-rigor option. The middle tier (Silver) is the
most common starting point.

## Migration paths

- **Self-attested → V-Certified Bronze**: drop your existing
  documentation into the artifact upload; the review board verifies
  and issues. Typical time: 5-10 business days.
- **V-Certified → ISO 42001 certification**: V-Certified evidence
  artifacts directly satisfy ISO 42001 clauses A.5-A.10 (per the
  crosswalk). Submit them to your ISO-accredited body as input.
- **V-Certified Silver → Gold**: add the change-mgmt artifact + at
  least one of RM / PRM records, engage a third-party auditor from
  our partner panel.

## FAQ

**Q: Will my customers accept a V-Certified attestation in lieu of
SOC 2?**
A: That depends entirely on your customers' procurement requirements.
Some Series A/B-stage customers will accept V-Certified Silver as a
"good enough" verifier; enterprise procurement will typically still
require SOC 2 Type 2. We'd suggest using V-Certified to bridge the
gap during the 6-12 months it takes to complete a full SOC 2 cycle.

**Q: Can I revoke a V-Certified attestation?**
A: Yes. Customer-initiated revocation is available at any non-terminal
status via `DELETE /v1/attestations/{id}` with a `revocation_reason`.
Multi-party revocation (triggered by the review board for material
misrepresentation per D12) lands in 31.G post-launch hardening.

**Q: What happens to the cert when I revoke?**
A: The leaf cert serial gets added to the public CRL within 24 h.
Verifiers chaining to the CA + checking the CRL will see the cert as
revoked. The attestation row stays publicly visible at status
"revoked" for transparency.
