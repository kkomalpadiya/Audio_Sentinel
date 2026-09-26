# Consensus safety test matrix

## What B7.2 adds

B7.2 adds the end-to-end safety matrix in
`tests/test_consensus_safety.py`. The tests pass privacy-minimized evidence through
the Phase 6 scorer, B7.1 agreement engine, and A7.2 final verification service.
They assert the score, supporting branches, conflicting branches, final outcome,
reason receipts, review flag, and local alert-candidate flag together.

This task changes no production thresholds or decision rules. It provides
regression protection for the existing A7.1, B7.1, and A7.2 behavior.

## Scenario coverage

The main matrix covers these 15 cases:

| Case | Expected safety behavior |
| --- | --- |
| Clean zero evidence | No action |
| Clean low positive score | Log |
| Medium score with low-confidence acoustic evidence | Review |
| Critical score immediately below acoustic support threshold | Review for insufficient agreement |
| Critical score exactly at acoustic support threshold | Local alert candidate |
| Low-confidence `no_speech` with VAD speech | No conflict from sub-threshold evidence |
| High-confidence `no_speech` with VAD and language | All involved branches conflict; review |
| Acoustic speech with zero VAD segments | Acoustic and speech conflict; review |
| Non-threatening acoustic speech with active risk language | Acoustic and language conflict; review |
| Language finding without accepted transcript | Speech and language conflict; critical score stays in review |
| Transcript requiring review | Otherwise eligible critical alert stays in review |
| Ambiguous language | Uncertainty remains neutral and reviewable |
| Missing acoustic evidence | Missing branch cannot be treated as safe |
| Acoustic-only consent with high acoustic score | Review with consent-limited receipt |
| Safe language beside high-confidence acoustic evidence | Language remains neutral and cannot add alert support |

## Exact boundaries and taxonomy

The acoustic boundary tests cover peak scores `0.0`, `0.849999`, `0.85`, and
`1.0`. Scores below `0.85` remain neutral with a low-confidence reason. The exact
`0.85` boundary and values above it support risk for a risk-bearing label.

The language taxonomy test covers all six categories. `no_concerning_match`,
`ambiguous`, and `context_suppressed` remain neutral. Only `distress`, `threat`,
and `weapon_reference` can support risk. A separate test confirms that speech
presence is always neutral by itself and never counts toward the two-branch alert
gate.

## False-alert invariants

Every non-alert matrix case must set `alert_candidate=false`. Critical-score
cases that do not become alerts must carry at least one explicit safety
explanation:

- `insufficient_alert_agreement` when fewer than two eligible branches support;
- `evidence_conflict_review_required` when branches contradict one another; or
- `risk_uncertainty_review_required` when transcript, language, or missing-data
  uncertainty blocks alerting.

Only the clean critical case with acoustic and language support, no conflict, no
missing branch, and no blocking uncertainty may become a local alert candidate.
The tests do not send notifications.

## Verification

Run the B7.2 suite:

```powershell
python -m pytest tests/test_consensus_safety.py -q
```

Run the complete Phase 7 suite:

```powershell
python -m pytest tests/test_consensus_contracts.py tests/test_consensus_rules.py tests/test_consensus_service.py tests/test_consensus_safety.py -q
```

Run the complete project suite:

```powershell
python -m pytest -q
```

## Next task

A7.3 will add a custom verification-model interface for future training while
preserving the deterministic Phase 7 service as the current production path.
