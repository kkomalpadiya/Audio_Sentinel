# Configurable risk scoring

## What B6.1 adds

B6.1 implements the deterministic scoring engine in
`src/audio_sentinel/risk_scoring.py`. It consumes an already validated A6.1
`RiskInputSet`, applies a bounded local JSON rule set, and returns a validated
`RiskAssessmentDocument`.

The built-in rules live in
`src/audio_sentinel/resources/risk-scoring-rules-v1.json`. Their exact bytes are
SHA-256 pinned in code, and every assessment records the rule-set ID, version,
format version, and artifact digest. A changed configuration therefore produces a
different assessment identity and cannot silently masquerade as the built-in
policy.

B6.2 adds the broader scenario matrix documented in
`docs/risk-scoring-tests.md`. It verifies the documented behavior through the
public scorer rather than testing private arithmetic helpers.

## Score calculation

The v1 engine adds three independently capped branch scores, then caps the total at
100:

```text
score = min(100, acoustic points + speech points + language points)
```

The built-in branch behavior is:

| Branch | Calculation | Branch cap |
| --- | --- | ---: |
| Acoustic | Highest configured event-label weight, plus 10 when a positively weighted event reaches the inclusive 0.85 confidence threshold | 60 |
| Speech | 2 when at least one speech segment exists, plus 5 per review-required transcript up to 10 review points | 12 |
| Language | Sum of configured finding-category weights | 50 |

Acoustic aggregation uses the maximum label weight so overlapping detections do not
multiply the score. Language findings are summed because separate accepted-text
findings may contribute distinct evidence, but the branch cap bounds repetition.
Review-required transcript points are capped before multiplication, so even a very
large contract-valid count cannot overflow numeric conversion.

The built-in active weights are:

| Acoustic label | Points | Language category | Points |
| --- | ---: | --- | ---: |
| `siren` | 10 | `distress` | 25 |
| `smoke_alarm` | 15 | `threat` | 40 |
| `glass_break` | 25 | `weapon_reference` | 35 |
| `crowd_panic` | 30 | `ambiguous` | 5 |
| `distress_speech` | 30 | `no_concerning_match` | 0 |
| `threatening_speech` | 40 | `context_suppressed` | 0 |
| `weapon_reference` | 40 |  |  |
| `gunshot` | 45 |  |  |
| `explosion` | 40 |  |  |

`ambient`, `no_speech`, `speech_present`, and `non_threatening_speech` acoustic
labels are fixed at zero. The rule contract also requires
`no_concerning_match` and `context_suppressed` language findings to remain at zero,
which prevents configuration drift from turning known non-risk evidence into risk.

## Human review and missing data

The built-in rule set requires human review when any of these conditions is true:

- the score is 25 or higher;
- a transcript is marked review-required;
- language evidence is ambiguous; or
- an input branch is marked missing.

Missing branches add no score. They are recorded with branch-specific reason codes
and `missing_data_review_required`, so absent evidence is not mistaken for evidence
of safety. Consent-limited speech and language branches are recorded as not
permitted or not applicable and do not create a missing-data penalty.

## Configuration loading

`load_risk_rule_set_bytes()` validates bounded JSON bytes and can require an exact
SHA-256 digest. `load_builtin_risk_rule_set()` loads the pinned package resource
without network or ML runtime dependencies. Callers can pass any validated
`LoadedRiskRuleSet` to `score_risk()`:

```python
from datetime import UTC, datetime

from audio_sentinel.risk_scoring import load_risk_rule_set_bytes, score_risk

loaded = load_risk_rule_set_bytes(custom_json_bytes, expected_sha256=expected_hash)
assessment = score_risk(inputs, rule_set=loaded, now=datetime.now(UTC))
```

Rule artifacts must include every acoustic label and language category in canonical
order, use finite non-negative weights, obey branch and total caps, and retain zero
weights for the protected non-risk categories. Unknown fields are rejected.

## Determinism and privacy

For the same inputs and exact rule artifact, the score, reasons, severity, review
decision, and assessment ID are deterministic. `created_at` is intentionally not
part of the identity. The engine does not load raw audio, transcript text, speaker
identity, consensus state, or alert data, and its stable failures do not expose
private evidence.

The checked-in rule schema is
`docs/schemas/v1/risk-rule-set.schema.json`. The assessment schema remains at
`docs/schemas/v1/risk-assessment.schema.json` and now requires rule-set provenance.

## Verification

Run the complete focused Phase 6 tests:

```powershell
python -m pytest tests/test_risk_contracts.py tests/test_risk_scoring.py tests/test_risk_integration.py tests/test_risk_scenarios.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A6.3 defines the separate future-facing trainable interface in
`docs/trainable-risk-model.md`. The deterministic v1 scorer remains the active
implementation and its validated scenario baseline is unchanged.

A6.4 will validate risk-score behavior and thresholds across the supported scoring
boundaries.

## Beginner-friendly explanation

The project now has an actual scoring calculator, not only a score format. Its
weights live in a small checked JSON file, each evidence branch has a maximum
contribution, and the final result can always show exactly which version of the
rules produced it.
