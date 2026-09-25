# Risk evidence integration

## What A6.2 adds

A6.2 implements `integrate_risk_inputs()` in
`src/audio_sentinel/risk_integration.py`. It converts validated Phase 3 acoustic,
Phase 4 speech, and Phase 5 language evidence into the privacy-minimized
`RiskInputSet` consumed by the B6.1 scoring engine.

The adapter does not calculate a risk score. It establishes that the supplied
artifacts belong to the same source, summarizes only the fields permitted by the
A6.1 contract, and records an explicit status for every evidence branch.

## Integration flow

Callers provide a trusted `RiskSource` plus any available evidence documents:

```python
from audio_sentinel.risk_integration import integrate_risk_inputs
from audio_sentinel.risk_scoring import score_risk

inputs = integrate_risk_inputs(
    source,
    acoustic=acoustic_evidence,
    speech=speech_evidence,
    language=language_evidence,
)
assessment = score_risk(inputs)
```

The source carries the authorized clip ID, consent ID, processing scope, sample
rate, and sample count. Evidence is optional because missing branches must remain
representable rather than causing the pipeline to report a false safe result.

## Branch conversion

| Branch | Risk input summary |
| --- | --- |
| Acoustic | Event labels, sample-exact spans, seconds, peak scores, source event indexes, event count, and maximum peak score. |
| Speech | Segment count, accepted transcript count, review-required transcript count, and maximum VAD score. |
| Language | Finding and segment IDs, categories, reason codes, sample spans, rule-match counts, and finding count. |

Each present branch receives an evidence reference containing the source document
type, evidence ID, and a SHA-256 hash of the complete normalized evidence document.
Transcript text, matched text, raw audio, model tensors, speaker data, and absolute
paths are not copied into risk inputs.

## Provenance checks

The adapter rejects evidence when:

- its clip ID, consent ID, sample rate, sample count, or processing scope conflicts
  with the risk source;
- acoustic and speech artifacts reference different prepared manifests or raw audio;
- language evidence does not reference the exact supplied speech evidence ID and
  document hash;
- language analyses do not exactly cover the accepted speech transcripts; or
- transcript IDs, spans, text hashes, byte/character counts, confidence scores, or
  confidence kinds differ between the speech and language artifacts.

These checks happen before a `RiskInputSet` is returned. Failures use stable,
non-sensitive error codes and do not echo transcript content.

## Missing and consent-limited branches

The adapter assigns statuses deterministically:

| Situation | Acoustic | Speech | Language |
| --- | --- | --- | --- |
| Evidence supplied and valid | `present` | `present` | `present` |
| Expected evidence absent | `missing` | `missing` | `missing` |
| Acoustic-only consent | As available | `not_permitted` | `not_permitted` |
| Speech present with zero accepted transcripts | As available | `present` | `no_accepted_text` |

An empty, valid language artifact may accompany a speech artifact with zero accepted
transcripts. The resulting language status is still `no_accepted_text`, and the
empty artifact is not presented as scored language evidence.

Supplying speech or language evidence under acoustic-only consent is rejected rather
than silently ignored. Supplying language evidence without its speech dependency is
also rejected.

## Verification

Run the focused A6.2 tests:

```powershell
python -m pytest tests/test_risk_integration.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

## Next task

B6.2 will add a broader risk-scoring scenario matrix covering normal combinations,
missing branches, consent boundaries, caps, and score-band edges.

## Beginner-friendly explanation

The scorer no longer needs someone to hand-build its input. This adapter checks
that the three evidence reports describe the same authorized audio clip, keeps only
the small summaries needed for scoring, and clearly labels anything missing or
disallowed.
