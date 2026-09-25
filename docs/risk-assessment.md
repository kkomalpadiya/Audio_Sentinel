# Risk assessment contract

## What A6.1 adds

A6.1 defines the first Phase 6 risk-assessment contract in
`src/audio_sentinel/risk_contracts.py`. It specifies the inputs that the later risk
scoring engine may consume, the fixed 0-100 score range, the v1 severity bands, and
the missing-data policy.

The checked-in JSON Schema is in `docs/schemas/v1/risk-assessment.schema.json`, and
the complete example is in `docs/examples/risk-assessment.json`.

This task defines the contract. B6.1 now implements the configurable scoring
formula described in `docs/risk-scoring.md`. Neither task trains a model, makes a
consensus decision, creates an alert, or claims that an incident occurred.

## Input boundary

Risk assessment may summarize only prior evidence artifacts:

| Branch | Source artifact | Purpose |
| --- | --- | --- |
| Acoustic | `acoustic_event_candidates` | Candidate non-speech and speech-like sound events from Phase 3. |
| Speech | `speech_evidence_candidates` | Speech presence and transcript reliability counts from Phase 4. |
| Language | `language_evidence` | Accepted-transcript language findings from Phase 5. |

Each branch records a status:

| Status | Meaning |
| --- | --- |
| `present` | The branch has a hash-pinned evidence reference and scoring summary. |
| `missing` | The branch was expected but no usable evidence is available. |
| `not_permitted` | Consent scope does not allow that branch. |
| `not_applicable` | The branch is not relevant for this source. |
| `no_accepted_text` | Speech ran, but no accepted transcript exists for language analysis. |

Non-present branches cannot carry scoring counts or signals. Present branches must
include the correct evidence kind, document type, evidence ID, and SHA-256 hash.

## Score and severity bands

The v1 score is always a finite number from 0 to 100. The default severity bands are:

| Severity | Score range |
| --- | --- |
| `none` | exactly `0` |
| `low` | greater than `0` through `24` |
| `medium` | `25` through `49` |
| `high` | `50` through `74` |
| `critical` | `75` through `100` |

The contract validates that the recorded severity matches the recorded score. Future
rules may change how the score is calculated, but they cannot silently change this
v1 score range or band mapping.

## Missing-data policy

Missing evidence is not treated as proof of safety. The default policy requires the
acoustic branch and records whether speech or language evidence is missing,
unavailable because of consent, or unavailable because there was no accepted text.

If a required or expected branch is missing, the risk document must record the
missing branch, include the matching reason code, and set human review when the
missing-data state requires review. Consent-limited branches use `not_permitted`
instead of being counted as missing.

## Privacy and scope

The risk document stores branch summaries, IDs, hashes, sample spans, labels, counts,
and reason codes. It does not copy raw audio, full transcript text, speaker identity,
speaker embeddings, consensus approval, alert IDs, or notification details.

The language branch stores finding IDs, segment IDs, categories, reason codes, sample
spans, and rule-match counts. It does not repeat the matched text or transcript text.

## Integrity rules

The contract checks that:

- evidence references match their branch kind and document type;
- present branches include evidence and non-present branches carry no scoring data;
- acoustic signal counts and peak scores match the signal inventory;
- speech transcript counts do not exceed speech segment counts;
- language findings are unique and require present speech provenance;
- acoustic and language signal spans stay within the source clip;
- acoustic timestamps equal sample offsets divided by sample rate;
- reason codes and missing branch lists use canonical ordering;
- missing-data summaries match the branch statuses; and
- the recorded severity matches the 0-100 score band.

## Verification

Run the focused contract tests:

```powershell
python -m pytest tests/test_risk_contracts.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

A6.2 will convert validated acoustic, speech, and language evidence documents into
the risk inputs consumed by the B6.1 engine.

## Beginner-friendly explanation

This task creates the receipt format for risk scoring. It says which evidence was
available, which evidence was missing, how a score must be labeled, and when missing
data forces human review. B6.1 now performs the scoring calculation using an
integrity-checked local rule file.
