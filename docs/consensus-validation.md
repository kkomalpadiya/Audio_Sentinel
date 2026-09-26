# End-to-end consensus outcome validation

## Decision

A7.4 validates the deterministic Phase 6-7 pipeline against 13 synthetic,
reviewable acceptance scenarios. All 13 pass, so the recorded decision is
`validated_for_deterministic_v1_scenarios`.

The generated report is
`outputs/a7_4_validation/consensus_pipeline_validation.json`. It records the
scenario-suite hash, exact risk-rule and agreement-rule hashes, consensus policy,
expected and observed result for every case, pass/fail counts, and a deterministic
validation ID.

## Executable boundary

The compact scenario corpus is
`src/audio_sentinel/resources/consensus-validation-scenarios-v1.json`. Its exact
bytes are SHA-256 pinned by `src/audio_sentinel/consensus_validation.py`. The suite
also pins the exact B6.1 risk rules, B7.1 agreement rules, and A7.1 policy version.
It cannot silently validate a different deterministic policy.

Each case expands into a complete public `RiskInputSet`. The validator then calls,
in order:

1. `score_risk()` to produce the Phase 6 assessment;
2. `evaluate_evidence_agreement()` to classify branch support and conflict; and
3. `decide_consensus()` to revalidate the handoff and select the A7.1 outcome.

The comparison covers score, severity, Phase 6 review state, all three branch
statuses and agreement states, supporting and conflicting branches, final outcome,
decision reasons, review state, and the local alert-candidate flag. It does not
call private policy helpers.

## Scenario coverage

| Scenario family | Acceptance evidence |
| --- | --- |
| All four outcomes | Clean `no_action`, `log`, `review`, and `alert` cases |
| Alert threshold | Acoustic confidence immediately below `0.85` stays neutral; exactly `0.85` supports risk |
| Multi-branch alert | Only clean critical acoustic and language support produces a local alert candidate |
| Single-branch block | A critical score with language support alone remains in review |
| Cross-branch conflicts | All four B7.1 conflict rules are exercised end to end |
| Uncertainty | Transcript-review uncertainty blocks an otherwise eligible alert |
| Missing evidence | Missing branches remain unavailable and force review instead of no action |
| Consent | Not-permitted speech and language remain explicit in an acoustic-only case |
| Safe language | `no_concerning_match` stays neutral and cannot count as alert support |

Only `critical-clean-alert` has `alert_candidate=true`. That result is still
`review_required=true`, is local only, and does not send a notification.

## Integrity and reproducibility

The report identity excludes `created_at`, so repeated runs using the same suite
and policy artifacts have the same `validation_id`. The output is
privacy-minimized: it excludes raw audio, transcript text, matched text, speaker
identity, file paths, and recipients.

The command refuses to overwrite an existing report:

```powershell
python scripts/validate_consensus_pipeline.py --output outputs/a7_4_validation/reproduced.json
```

For an exact comparison, the reproduced report should have the same
`validation_id`; only `created_at` may differ.

## Meaning and limitation

This result approves the deterministic v1 mechanics for the checked synthetic
scenarios. It does not establish real-world sensitivity, specificity,
false-positive rate, false-negative rate, model accuracy, incident accuracy, or
notification reliability. Those claims require representative labeled recordings
and later operational evaluation.

The report therefore carries the explicit limitation
`synthetic_acceptance_validation_not_empirical_incident_accuracy`.

## Verification

Run the focused A7.4 tests:

```powershell
python -m pytest tests/test_consensus_validation.py -q
```

Run the complete Phase 7 suite:

```powershell
python -m pytest tests/test_consensus_contracts.py tests/test_consensus_rules.py tests/test_consensus_service.py tests/test_consensus_safety.py tests/test_verification_model.py tests/test_consensus_validation.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A8.1 will build offline evaluator orchestration for one recorded clip. A7.4
validates deterministic decision mechanics, but deliberately does not add
recording orchestration, notification delivery, or empirical model evaluation.
