# Risk-score behavior and threshold validation

## Decision

A6.4 validates the exact built-in deterministic risk policy against 13 synthetic,
reviewable acceptance scenarios. All 13 pass, so the recorded decision is
`validated_for_deterministic_v1_scenarios`.

The generated report is
`outputs/a6_4_validation/risk_score_validation.json`. It records the validation
suite hash, scoring-rule hash, threshold snapshot, expected and observed outcome
for every case, pass/fail counts, and a deterministic validation ID.

## Executable boundary

The compact scenario corpus is
`src/audio_sentinel/resources/risk-validation-scenarios-v1.json`. Its exact bytes
are SHA-256 pinned by `src/audio_sentinel/risk_validation.py`, and the suite itself
pins the exact B6.1 rule artifact it targets. A suite cannot silently run against a
different scoring policy.

Each compact case is expanded into a complete `RiskInputSet` through the public
A6.1 contract. The runner then calls the public `score_risk()` function and compares
the resulting score, severity, review decision, reason codes, and missing branches
with the expected outcome. It does not call private scoring helpers.

## Validated behaviors

The suite covers:

| Behavior | Acceptance evidence |
| --- | --- |
| Severity mapping | Representative `none`, `low`, `medium`, `high`, and `critical` outcomes |
| Acoustic confidence | A signal immediately below 0.85 gets no bonus; a signal exactly at 0.85 gets the bonus |
| Score review | A score below 25 does not trigger score-based review; a score exactly at 25 does |
| Branch caps | Speech stops at 12, language stops at 50, and acoustic cap metadata is recorded |
| Total cap | Strong combined evidence stops at 100 |
| Additional review | Review-required transcripts and ambiguous language require review below score 25 |
| Safe context | Context-suppressed language adds zero language points and does not independently require review |
| Missing evidence | Missing branches add no points, remain named, and require review |
| Consent | Not-permitted speech/language branches are not mislabeled as missing |
| No accepted text | Language absence after speech with no accepted transcript is not treated as missing evidence |

The threshold snapshot in the passing report records:

| Policy value | Built-in value |
| --- | ---: |
| Acoustic high-confidence threshold | 0.85 |
| Human-review minimum score | 25 |
| Acoustic branch cap | 60 |
| Speech branch cap | 12 |
| Language branch cap | 50 |
| Total score cap | 100 |

The detailed exact score-band boundaries and custom-rule edge cases remain covered
by the B6.2 regression matrix in `tests/test_risk_scenarios.py`.

## Meaning and limitation

This result approves the deterministic v1 policy's documented mechanics for the
checked scenarios. It does not establish real-world sensitivity, specificity,
false-positive rate, false-negative rate, or incident-prediction accuracy. Those
claims require representative labeled end-to-end evaluation data in Phase 9.

The report therefore carries the explicit limitation
`synthetic_acceptance_validation_not_empirical_incident_calibration`. Phase 7 may
use the validated risk assessment as one input, but must not reinterpret the score
as an automatically confirmed incident or alert.

## Reproduce the report

The CLI writes a new report and refuses to overwrite an existing file:

```powershell
python scripts/validate_risk_policy.py --output outputs/a6_4_validation/reproduced.json
```

For an exact comparison, the reproduced report should have the same
`validation_id`; only `created_at` may differ.

Run the focused validation tests:

```powershell
python -m pytest tests/test_risk_validation.py -q
```

Run all Phase 6 tests:

```powershell
python -m pytest tests/test_risk_contracts.py tests/test_risk_scoring.py tests/test_risk_integration.py tests/test_risk_scenarios.py tests/test_risk_model.py tests/test_risk_validation.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A7.1 will define the consensus policy and outcomes: no action, log, review, and
alert. Consensus will consume, not rewrite, the validated Phase 6 assessment.
