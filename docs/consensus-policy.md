# Consensus policy and outcomes

## What A7.1 adds

A7.1 defines the Phase 7 consensus contract in
`src/audio_sentinel/consensus_contracts.py`. It consumes a hash-pinned snapshot of
one Phase 6 risk assessment plus one typed agreement state for each acoustic,
speech, and language branch. It produces exactly one primary outcome:
`no_action`, `log`, `review`, or `alert`.

The checked-in JSON Schema is
`docs/schemas/v1/consensus-decision.schema.json`, and a complete critical example
is in `docs/examples/consensus-decision.json`.

This task defines the contract and safety policy. B7.1 will calculate branch
agreement and conflicts. A7.2 will implement the decision service. No alert is
created, delivered, or sent in A7.1.

## Outcome policy

The four outcomes are mutually exclusive primary actions:

| Outcome | V1 meaning |
| --- | --- |
| `no_action` | The Phase 6 score is exactly 0 and there is no support, conflict, missing evidence, or review trigger. |
| `log` | The score is greater than 0 through 24 and there is no review trigger. The later evaluator may retain the decision in its local report. |
| `review` | A score of at least 25, Phase 6 uncertainty, missing evidence, an evidence conflict, or insufficient agreement prevents automatic escalation. |
| `alert` | A local alert candidate may be created later because every v1 alert gate passed. It is not notification delivery. |

Outcome precedence is `alert`, `review`, `log`, then `no_action`. Alert eligibility
is checked first so a clean, corroborated critical case may become an alert
candidate even though Phase 6 correctly marks every score at or above 25 for human
review. An alert outcome still records `review_required: true`; the alert candidate
does not erase the Phase 6 review requirement.

## Alert safety gates

V1 permits an alert outcome only when all of these conditions hold:

- the Phase 6 score is at least 75 and the severity is `critical`;
- both alert-eligible branches, acoustic and language, support the risk assessment;
- no evidence branch conflicts with the assessment;
- no expected evidence branch is missing; and
- Phase 6 did not record transcript uncertainty, ambiguous language, or a
  missing-data review reason.

Speech presence alone is not risk support. It may be neutral or identify a
conflict, but ordinary speech is not enough to help authorize an alert. Language
support is accepted only as a typed branch state; B7.1 will own the exact rules
that decide whether a finding supports or conflicts.

Consent-limited branches remain `not_permitted`, not `missing`. In the v1 policy an
acoustic-only assessment cannot satisfy the two alert-support gates, so it may be
logged or reviewed but cannot silently become an alert.

## Branch agreement states

Every decision contains acoustic, speech, and language states in canonical order.
The agreement value must match the Phase 6 branch status:

| Phase 6 status | Allowed consensus state |
| --- | --- |
| `present` | `supports_risk`, `neutral`, or `conflicts_risk` |
| `missing` | `unavailable` |
| `not_permitted` | `not_permitted` |
| `not_applicable` or `no_accepted_text` | `not_applicable` |

The branch inventory must match the missing and not-permitted branch lists copied
from the hash-pinned risk snapshot. This prevents a decision from hiding missing
evidence by relabeling it as neutral.

## Integrity, privacy, and scope

The contract records the risk assessment ID, canonical SHA-256 hash, clip ID,
score, severity, risk reasons, branch statuses, consensus reasons, policy version,
and decision flags. It does not copy transcript text, matched phrases, raw audio,
speaker identity, contact information, notification targets, or delivery status.

`alert` means only that the later Phase 8 alert builder may create a local alert
record. This contract does not contact emergency services, send email or SMS, or
claim that a real incident has been empirically confirmed.

## Verification

Run the focused contract tests:

```powershell
python -m pytest tests/test_consensus_contracts.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

B7.1 will implement the evidence-agreement and conflict-detection rules that
produce the three typed branch states consumed by this contract.

## Beginner-friendly explanation

The risk score is one opinion. Consensus is the safety gate that decides what the
system should do with it. A quiet, complete result can do nothing; a small signal
can be logged; uncertain or conflicting evidence goes to review; and only a clean
critical result supported by both sound and language can become a local alert
candidate.
