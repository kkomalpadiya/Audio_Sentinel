# Speech transcription orchestration and confidence handling (A4.3)

## Outcome

`audio_sentinel.speech_transcription` connects the transcript-free A4.2 result to
the B4.2 offline transcriber. It reconstructs each sample-exact speech segment from
verified prepared windows, transcribes every segment independently, applies the
recorded A4.1 reliability policy, and produces a completed
`SpeechEvidenceDocument`.

The result also has a separate `downstream_transcripts` collection. Only candidates
whose assessment is `accepted` appear there. Rejected, review-required, and empty
results remain visible in evidence but cannot silently flow into later automatic
language analysis.

## Verified handoff

`orchestrate_speech_transcription()` accepts:

- project `Paths`;
- an in-memory `SpeechSegmentationResult` from A4.2;
- the verified B4.2 `LoadedTranscriptionModel`;
- optional orchestration and model resource settings; and
- an optional timezone-aware processing time.

Before reading audio, it validates the pinned transcription model and the complete
A4.2 snapshot. The public evidence and diagnostic window/segment inventories must
agree on IDs, paths, hashes, sample spans, timestamps, VAD scores, counts, model
provenance, and semantic evidence identity. The input must still be transcript-free;
already-enriched or tampered evidence is rejected.

The orchestrator then reopens the prepared manifest under `data/interim`, verifies
its SHA-256 identity and source metadata, and requires current
`acoustic_and_speech` consent. Selected window records must exactly match A4.2.
Every window is reread, size-limited, hash-checked, and decoded as the recorded
16 kHz mono PCM-16 WAV with only verified zero padding removed.

## Exact segment reconstruction

Speech intervals may cross multiple overlapping prepared windows. For each final
A4.2 interval, the orchestrator copies only the absolute sample intersection from
every referenced window into one contiguous float32 waveform. Every sample must be
covered. Where windows overlap, their decoded samples must be exactly equal; a
difference fails as a source mismatch rather than choosing one silently.

Each reconstructed waveform is passed to B4.2 as an independent model call. No
text, prompt, decoder history, or waveform from another speech interval is carried
forward. This preserves the segment boundaries and deterministic decoding contract.

The manifest, consent, and all used window hashes are checked again after inference.
Changes during the run invalidate the complete operation; no partial evidence is
returned.

## Low-confidence handling

The orchestrator calls `SpeechReliabilityPolicy.assess()` for every model candidate.
The default A4.1 boundaries behave as follows:

| Model result | Recorded outcome | Automatic downstream text |
| --- | --- | ---: |
| No text candidate | `not_transcribed` | No |
| Derived score below 0.50 | `rejected_low_confidence` | No |
| Derived score from 0.50 to below 0.80 | `review_required` | No |
| Derived score at or above 0.80 | `accepted` | Yes |

The complete evidence retains nonempty candidates at every confidence level so an
auditor can see what the model proposed and why it was blocked. The separate
`DownstreamTranscript` type contains only accepted text plus exact timestamps and
the typed confidence signal. It is the only automatic language-analysis handoff.
Missing text is not interpreted as benign speech, and review-required text remains
blocked until an approved human-review workflow exists.

## Evidence and identity

The completed evidence preserves the original source, VAD model, windows, segment
times, and policy, and adds:

- pinned transcription-model provenance;
- each optional `TranscriptCandidate`;
- the policy-recomputed assessment for every segment; and
- an exact count of segments with nonempty candidates.

Its semantic SHA-256 identity includes the transcript candidates, assessments, and
transcription model but excludes creation time. Equivalent reruns therefore retain
the same ID. The orchestration result is immutable and JSON-ready and contains no
waveform or absolute filesystem path.

## Resource and privacy boundaries

`SpeechOrchestrationSettings` bounds manifest and window bytes, decoded memory,
segment count, total transcribed samples, and accepted-text handoff bytes. The B4.2
settings independently bound each model input and output. Exceeding any limit fails
the whole operation.

The result contains transcript text, so it remains consent-controlled speech
evidence. It does not store raw audio, speaker identity or embeddings, classify the
meaning of the text, create an incident, or assign risk.

## Verification

Run the focused tests:

```powershell
python -m pytest tests/test_speech_transcription.py -q
```

Run the real pinned-model structural check:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_speech_transcription.py
```

The real check prepares a generated 1.6-second clip, forces complete structural VAD
coverage with a zero test threshold, reconstructs the resulting interval from three
overlapping windows, and runs the pinned offline transcriber twice. The tone yields
no text, and both runs produce identical evidence with an explicit
`not_transcribed` outcome and no downstream handoff. Positive text and every
confidence band are covered by isolated deterministic tests.

On a fresh machine, `.\scripts\setup_speech.ps1` installs both pinned local models
and runs the VAD, extraction, transcription-wrapper, and orchestration smoke tests.

## Beginner-friendly explanation

This step is the gatekeeper between speech recognition and later language analysis.
It rebuilds each detected speech interval from proven audio, asks the offline model
for words, and applies the recorded confidence rules. The model's guess stays in
the receipt, but only a sufficiently reliable guess enters the automatic downstream
list. Uncertain or missing words stop here instead of quietly influencing an alert.
