# Speech evidence contract and reliability rules

## What A4.1 adds

A4.1 defines the handoff format for the speech-processing branch before a VAD or
transcription model is connected. The public Python contract is in
`src/audio_sentinel/speech_contracts.py`; the portable JSON Schema is
`docs/schemas/v1/speech-evidence.schema.json`; and a complete illustrative document
is in `docs/examples/speech-evidence.json`.

This task defines evidence and handling rules. It does not detect speech, transcribe
audio, classify language, assign risk, or persist evidence. Those behaviors remain
separate later tasks.

## Contract boundary

A `SpeechEvidenceDocument` records:

- a version and document type;
- an opaque evidence ID and timezone-aware creation time;
- the approved prepared-manifest path and hashes that identify the source audio;
- the consent reference and the required `acoustic_and_speech` processing scope;
- the source sample rate and exact prepared-sample count;
- the reliability policy used for the run;
- VAD model provenance and optional transcription-model provenance;
- the complete prepared-window inventory supplied to the branch; and
- chronologically ordered, non-overlapping speech segments with sample-exact times,
  VAD evidence, an optional transcript candidate, and a verified reliability
  assessment.

The document stores text and provenance, but no raw audio, speaker name, speaker
embedding, language-intent category, incident outcome, severity, or risk score.
Speech evidence is therefore not an alert and cannot make a risk decision by itself.

## Reliability policy v1.0

The default policy has three explicit settings:

| Setting | Default | Boundary rule |
| --- | ---: | --- |
| VAD speech threshold | 0.60 | A recorded speech segment must have `vad_score >= 0.60`. |
| Transcript review threshold | 0.50 | Scores below 0.50 are rejected as low-confidence. |
| Transcript acceptance threshold | 0.80 | Scores from 0.50 to below 0.80 require review; scores at or above 0.80 are accepted. |

These values are versioned starting points, not measured production calibration.
The VAD and transcription evaluation tasks must test them with labeled data before
deployment. A score is only a normalized evidence signal unless its
`confidence_kind` says it is a calibrated probability.

The assessment states both the reliability label and the permitted action:

| Reliability | Required reason | Downstream text allowed | Human review required |
| --- | --- | ---: | ---: |
| `not_transcribed` | `transcript_not_attempted` | No | No |
| `rejected_low_confidence` | `below_review_threshold` | No | No |
| `review_required` | `below_acceptance_threshold` | No | Yes |
| `accepted` | `meets_acceptance_threshold` | Yes | No |

Only `accepted` transcript text may be handed automatically to the later language
analysis branch. `review_required` is deliberately blocked until a person or a later
approved review workflow resolves it. Missing transcription is not converted into
harmless speech, and a rejected hypothesis is not treated as proof that no threat was
spoken.

The document validator recomputes every assessment from the recorded policy and
transcript score. A producer cannot mark a low-confidence transcript as accepted by
changing the label or Boolean flags.

## Time and source integrity

Sample offsets are authoritative. `start_seconds` and `end_seconds` must equal the
sample offsets divided by the source sample rate to within one nanosecond. Every
segment must:

- fall inside the prepared source;
- overlap every window it cites, with the cited windows jointly covering its full
  span without gaps;
- cite known, unique window IDs;
- meet the recorded VAD threshold; and
- appear in deterministic chronological order without overlapping another final
  speech segment.

Windows and segments have unique IDs and bounded portable paths. Summary counts must
match their inventories. If any segment contains a transcript, the document must
also record the transcription model and runtime provenance.

## Privacy and consent boundary

The speech contract requires `processing_scope: acoustic_and_speech`. An
`acoustic_only` or inactive consent record must be rejected before this document is
created. The contract carries only the consent ID; the full consent record remains in
the prepared-audio manifest rather than being copied into every evidence artifact.

The schema forbids unknown fields. This prevents accidental inclusion of raw audio,
speaker identity, embeddings, or later-stage decisions without a deliberate schema
revision and review.

## Verification

Run the focused contract tests:

```powershell
python -m pytest tests/test_speech_contracts.py -q
```

Run the complete project checks before committing:

```powershell
.\scripts\verify_project.ps1
```

The focused suite checks both sides of every reliability threshold, missing
transcripts, policy tampering, VAD gating, consent scope, model provenance, exact
time conversion, ordering, overlap, source-window references, path safety,
immutability, schema export, and the checked-in example.

## Next task

B4.1 will implement the pretrained voice-activity-detection wrapper. It must emit
normalized VAD scores and model provenance that satisfy this contract; it must not
invent transcripts or risk outcomes.

## Beginner-friendly explanation

This contract is a strict receipt for the speech branch. It says where a possible
spoken segment came from, what the speech detector scored, what text (if any) the
transcriber suggested, and whether that text is safe to pass forward automatically.
Uncertain words are kept visibly uncertain instead of being silently trusted or
silently treated as harmless.
