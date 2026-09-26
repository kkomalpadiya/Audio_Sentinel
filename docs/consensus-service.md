# Final verification decision service

## What A7.2 adds

A7.2 implements the Phase 7 decision service in
`src/audio_sentinel/consensus_service.py`. It accepts one validated Phase 6
`RiskAssessmentDocument` and its B7.1 `ConsensusAgreementEvaluation`, verifies
that they belong together, and returns exactly one A7.1
`ConsensusDecisionDocument` outcome:

- `no_action`
- `log`
- `review`
- `alert`

The `alert` outcome is only a local alert candidate. This service does not create
an external incident, contact a recipient, or send a notification.

## Verification before decision

The service fails closed before applying the policy. It:

1. revalidates the complete Phase 6 assessment and B7.1 evaluation;
2. recomputes the assessment's canonical SHA-256 and checks both assessment
   identity fields in the agreement handoff;
3. reruns B7.1 with the trusted hash-pinned rule artifact and the original
   evaluation timestamp; and
4. requires the recomputed result to equal the supplied evaluation exactly.

The built-in B7.1 rule set is trusted by default. A caller may use a custom rule
artifact only by passing the exact validated `LoadedAgreementRuleSet` to
`FinalVerificationService` or `decide_consensus`. Merely claiming a custom rule
identifier or checksum in the evaluation is insufficient.

Failures use `ConsensusVerificationError` with stable codes such as
`invalid_assessment`, `invalid_agreement`, `assessment_mismatch`,
`untrusted_agreement`, and `invalid_time`. Messages do not include transcript or
audio contents.

## Outcome precedence

The service applies the A7.1 precedence order.

### Alert

A local alert candidate requires every gate below:

- score at or above 75 with `critical` severity;
- support from both alert-eligible branches: acoustic and language;
- no conflicting branch;
- no missing evidence branch; and
- no transcript-review, ambiguous-language, or missing-data uncertainty reason.

The decision records `critical_risk`, `multi_branch_alert_agreement`, and
`alert_candidate`. It also sets `review_required=true`; later alert handling must
not interpret this local candidate as delivery authorization.

### Review

Review takes precedence over logging and no action when the Phase 6 score or
review flag requires it, or when evidence is missing or conflicting. The decision
records the applicable receipts for score threshold, uncertainty, missing
evidence, conflict, and insufficient multi-branch agreement. A critical score
that fails an alert gate stays in review.

### Log and no action

A clean score from 1 through 24 is logged. A zero score with no support, conflict,
missing evidence, or review trigger selects no action. Consent-limited branches
remain explicit through the `consent_limited_evidence` receipt; they are not
silently converted into present evidence.

## Determinism and privacy

The decision ID is the SHA-256 of canonical policy, risk reference, branch state,
outcome, and reason data. The timestamp is deliberately excluded, so rerunning
the same verified evidence yields the same decision identity. The output retains
only IDs, hashes, enum states, scores, and reason codes. It does not contain raw
audio, transcript text, matched text, speaker identity, paths, or recipient data.

## Usage

```python
from audio_sentinel.consensus_rules import evaluate_evidence_agreement
from audio_sentinel.consensus_service import decide_consensus

agreement = evaluate_evidence_agreement(assessment)
decision = decide_consensus(assessment, agreement)
```

No network access or model download is required.

## Verification

Run the focused Phase 7 suite:

```powershell
python -m pytest tests/test_consensus_service.py tests/test_consensus_rules.py tests/test_consensus_contracts.py -q
```

Run the complete project suite:

```powershell
python -m pytest -q
```

## Safety coverage

B7.2 adds the [consensus safety test matrix](consensus-safety-tests.md) for
disagreement, exact confidence boundaries, missing and consent-limited evidence,
and false-alert prevention. A7.3 is the next task.
