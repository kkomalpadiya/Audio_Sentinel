# Final privacy, security, and deployment review

## Executive decision

A10.4 is complete. The reviewed build is **not approved for production**. Its
allowed use remains controlled local research and offline evaluation with
authorized recordings and trained human review.

The review found 17 items across privacy, security, and deployment:

| Disposition | Count | Meaning |
| --- | ---: | --- |
| Verified control | 5 | The repository contains executable and tested evidence for the stated boundary. |
| Conditional control | 2 | A useful control exists, but its protection is narrower than production needs. |
| Release blocker | 10 | Production deployment must not proceed until the required action is implemented and independently approved. |

This decision is intentionally conservative. Passing tests, deterministic hashes,
and a successful synthetic performance run do not establish lawful recording,
field accuracy, secure operations, or a safe real-world alerting service.

## Review method

The executable review in `src/audio_sentinel/deployment_review.py` uses a fixed
scope rather than accepting caller-selected findings. It:

1. loads 16 repository evidence artifacts covering policy, implementation, tests,
   the held-out evaluation, and the A10.3 live-path measurement;
2. resolves every file beneath the supplied project root and rejects missing,
   empty, oversized, or escaping evidence;
3. records each evidence file's normalized relative path, byte size, purpose, and
   SHA-256 digest;
4. assigns every finding a domain, disposition, severity, evidence references,
   conclusion, and—when it blocks release—a concrete required action;
5. recomputes the inventory counts and canonical review identity; and
6. writes a bounded, schema-validated JSON document atomically, with replacement
   permitted only when explicitly requested.

`verify_review_evidence` can later rehash the repository files and identify which
evidence has changed since the review. A matching hash detects content drift; it
does not prove who authored or approved a file.

## Privacy conclusions

### Verified controls

- Consent distinguishes denied, withdrawn, acoustic-only, and speech-authorized
  processing, and current authorization is checked at live boundaries.
- Live PCM is bounded in memory. Delivery and retry are explicit, and discarded
  queue copies are zeroed rather than persisted.
- Final reports and the final review omit raw audio, transcript text, credentials,
  recipients, and absolute paths.

### Release blockers

- **PRIV-04 — incomplete lifecycle coverage.** Automated cleanup covers only
  verified final bundles. Raw recordings, intermediate speech text, backups,
  exports, and secure erasure still require a complete policy-owned inventory and
  enforcement mechanism.
- **PRIV-05 — no organizational privacy approval.** Software checks cannot prove
  lawful recording authority, authentic consent, required notice, purpose
  limitation, data-subject rights, or jurisdiction-specific compliance.

## Security conclusions

### Verified or conditional controls

- The HTTP evaluator is loopback-only and accepts no upload, URL, recipient,
  transport setting, or notification authority.
- Live sessions bind revocable enrollment, one-time HMAC challenges, single-use
  receipts, current consent, exact PCM geometry, and ordered chunks.
- Strict schemas, bounded reads, canonical hashes, and atomic writes provide
  useful integrity protection. This is conditional because an ordinary hash is
  neither an author signature nor a trusted chain of custody.

### Release blockers

- **SEC-04 — credential lifecycle.** The caller supplies the HMAC secret resolver;
  the project has no approved keystore, authenticated provisioning, rotation,
  recovery, or compromise-response implementation.
- **SEC-05 — host trust boundary.** Loopback does not protect against another local
  user or a compromised process. OS identities, ACLs, encryption, isolation, and
  endpoint hardening remain outside the application.
- **SEC-06 — independent assurance.** No independent threat model, penetration
  test, dependency-vulnerability release gate, software bill of materials, or
  signed-release evidence exists.

## Deployment conclusions

### Conditional evidence

The A10.3 regression successfully exercised fresh authentication, bounded
buffering, one deferred retry, backpressure rejection, rolling windows, decision
timing, and a local-alert timing probe. It reconciled 12 accepted chunks, five
processed windows, zero loss, zero duplicate acceptance, and zero sequence gaps.

That evidence is deliberately classified as conditional. It was accelerated,
synthetic, and in-process, with only one connection and one alert timing sample.

### Release blockers

- **DEP-02 — missing production live path.** There is no microphone/driver capture
  adapter, production transport, supervised background service, or approved live
  model runner.
- **DEP-03 — nonrepresentative performance evidence.** Real hardware scheduling,
  network behavior, production inference, concurrency, recovery, soak duration,
  and stable tail latency are unmeasured.
- **DEP-04 — no reviewer or notification operations.** Alerts are local candidates;
  reviewer authentication, assignment, disposition, escalation, delivery, and
  external action are not implemented or authorized.
- **DEP-05 — no production operations package.** Observability, service-level
  objectives, incident response, disaster recovery, release signing, rollback,
  ownership, and support procedures are absent.
- **DEP-06 — insufficient detection validation.** The small balanced acoustic run
  is not representative of field prevalence, environments, speech, fairness,
  real incidents, or alert effectiveness.

## Production release gates

Before anyone revisits a production decision, the owner must close all ten release
blockers and obtain independent approval. At minimum, that requires:

1. complete privacy/legal authorization and end-to-end data lifecycle controls;
2. managed credentials, authenticated operators, host hardening, encryption, and
   an independent security assessment with supply-chain controls;
3. the intended capture, transport, model, reviewer, and notification path;
4. representative validation, calibration, human-factors testing, long-running
   performance and recovery evidence, and predeclared acceptance criteria; and
5. production monitoring, incident response, release, rollback, recovery,
   ownership, and support procedures.

Closing a blocker changes the implementation or evidence. Generate a new review;
do not edit the existing JSON outcome by hand.

## Review artifact

The checked-in result is
`outputs/a10_4_review/final-deployment-review.json`, and its portable contract is
`docs/schemas/v1/final-deployment-review.schema.json`. The document permanently
records `production_approved=false`, `notification_delivery_authorized=false`, and
`external_action_authorized=false`.

Regenerate it after an intentional evidence change:

```powershell
python scripts\run_final_deployment_review.py --replace
```

Run its focused verification:

```powershell
python -m pytest tests\test_deployment_review.py -q
```

No model download, microphone, network endpoint, notification destination, or
external service is needed to conduct this review.
