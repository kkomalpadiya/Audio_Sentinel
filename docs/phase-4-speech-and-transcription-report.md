# Phase 4 completion report: Speech and Transcription

## 1. What this phase accomplished

Phase 4 built the speech-processing branch of Project 1. It takes the authorized,
prepared audio produced in Phase 1, finds the parts that probably contain speech,
turns those parts into English text candidates with a local transcription model,
and decides which text is reliable enough to continue automatically to the future
language-analysis phase.

The phase was designed around one important safety rule: a model's guess is
evidence, not proof. Detecting speech does not identify the speaker, and producing
text does not prove that the text is correct, threatening, harmless, or connected
to an emergency. Phase 4 records what the models produced, how reliable the result
was considered, and why a result was accepted, held for review, rejected, or left
untranscribed.

At the end of this phase:

- all seven Speech and Transcription tasks are complete;
- 37 of the project's 72 tracked tasks are complete;
- the full project suite contains 1,034 passing tests;
- the exact local Silero voice-activity model and its runtime contract are verified;
- the exact local Faster-Whisper English transcription model is verified;
- speech-like frames are converted into clean, sample-exact speech intervals;
- overlapping audio windows do not duplicate or inflate the same speech evidence;
- each speech interval is reconstructed and transcribed independently;
- only accepted text may enter the later automatic language-analysis branch;
- uncertain, rejected, and missing text remains visible but blocked;
- final speech evidence is deterministic, JSON-ready, and tied to its source; and
- the next task is A5.1, which begins the Language Understanding phase.

## 2. The complete Phase 4 flow in common terms

```text
Authorized recording
        |
        v
Verified 16 kHz mono audio windows from Phase 1
        |
        v
Local Silero model scores tiny pieces for speech-likeness
        |
        v
Qualifying pieces from overlapping windows are placed on one timeline
        |
        v
Repeated pieces are merged into clean, non-overlapping speech intervals
        |
        v
Each interval is rebuilt from verified audio samples
        |
        v
Local Faster-Whisper model proposes English text
        |
        v
Recorded reliability rules assess each text candidate
        |
        +---- rejected text stays in evidence but stops
        |
        +---- uncertain text is marked for review and stops
        |
        +---- missing text is marked not transcribed and stops
        |
        +---- accepted text enters the downstream transcript list
        |
        v
Portable speech-evidence document for later language analysis
```

The system uses two separate model steps because they answer different questions:

1. **Voice activity detection:** "Does this short piece sound like speech?"
2. **Transcription:** "What English words might have been spoken?"

Keeping these jobs separate makes the behavior easier to test, explain, and audit.

## 3. Work completed task by task

| Task | Main result | Common-language meaning |
| --- | --- | --- |
| A4.1 | Speech-evidence schema and reliability rules | Defined the receipt format and the rules for what may continue. |
| B4.1 | Verified Silero VAD wrapper | Added a local model that scores speech-likeness in very short audio frames. |
| A4.2 | Speech-segment extraction | Combined qualifying frames into exact speech intervals without double counting overlaps. |
| B4.2 | Verified Faster-Whisper wrapper | Added a local English model that proposes text for one speech interval at a time. |
| A4.3 | Transcription orchestration | Connected exact speech intervals to transcription and enforced confidence handling. |
| B4.3 | Wrapper boundary tests | Tested both model adapters against normal, malformed, changed, and oversized inputs. |
| A4.4 | Full branch integration tests | Proved that the complete chain works across component boundaries and fails safely. |

### A4.1 — Speech evidence and reliability rules

The first task defined the contract that every later Phase 4 component must follow.
This contract is implemented in `src/audio_sentinel/speech_contracts.py` and has a
checked-in JSON Schema plus a complete example document.

In common terms, the contract is a strict receipt. It records:

- which authorized recording was used;
- which prepared-audio manifest and windows were used;
- which VAD and transcription models produced the results;
- which versions and runtimes were involved;
- the exact start and end of every speech interval;
- each optional text candidate and its confidence signal;
- the reliability rule applied to that candidate; and
- the final handling decision.

The contract deliberately does not contain a language-intent category, incident
decision, severity, or risk score. Those belong to later phases. Keeping them out
prevents a low-level speech model from quietly making a high-level safety decision.

The default version 1.0 reliability policy established three boundaries:

| Evidence | Boundary | Result |
| --- | ---: | --- |
| VAD speech score | At least 0.60 | The frame may support a speech interval. |
| Transcript score | Below 0.50 | Reject the text as too uncertain. |
| Transcript score | 0.50 to below 0.80 | Keep it for review, but block automation. |
| Transcript score | At least 0.80 | Accept it for downstream language analysis. |
| No text | No score | Mark it `not_transcribed`; never assume it was harmless. |

These numbers are recorded policy values, not universal truths. The code supports a
different versioned policy when future evaluation justifies one.

Confidence signals are also labeled by type. A raw or mathematically derived model
score is not described as a proven probability of correctness. This avoids giving
the number more meaning than it actually has.

Validation is intentionally strict. It rejects unknown fields, invalid numbers,
unsafe paths, missing provenance, incorrect counts, overlapping final segments,
incorrect timestamp conversions, unapproved processing scope, broken source-window
links, or decisions that do not match the recorded policy.

This task added 43 focused tests. They covered the policy boundaries, tampering,
timestamps, ordering, source links, consent, transcript limits, immutability, schema
generation, and privacy exclusions.

### B4.1 — Verified local voice-activity detection

The second task added `src/audio_sentinel/vad.py`, a narrow wrapper around Silero
VAD v6. VAD means voice activity detection. It does not understand the words. It
only estimates how speech-like each short piece of audio is.

The model is taken from the pinned `faster-whisper==1.2.1` distribution and runs
through pinned `onnxruntime==1.23.2` on the CPU. The wrapper accepts 16 kHz mono
`float32` audio, matching the output of the existing preparation pipeline.

Before loading the model, the code verifies:

- the path remains inside the approved local model directory;
- links and junctions are not redirecting the path;
- the expected model file and completion marker exist;
- no unexpected files were added;
- file sizes remain within the configured limit;
- the model's SHA-256 fingerprint still matches the pinned bytes;
- the installed runtime version is exactly the expected version;
- the CPU execution provider is available; and
- every ONNX input and output still has the expected name, type, and shape.

A SHA-256 fingerprint works like a digital seal. If the model changes by even one
byte, the fingerprint changes and loading stops.

The VAD examines 512 real samples at a time, which is 32 milliseconds at 16 kHz.
Silero also needs the preceding 64 samples as context. The wrapper constructs that
context explicitly, carries the model's short-term memory only within one waveform,
and resets it before the next independent waveform.

If the final frame is shorter than 512 samples, it is temporarily padded on the
right. The result still reports where the real audio ended, so artificial padding
cannot become speech evidence.

The wrapper returns raw speech probabilities and portable model metadata. It does
not create final speech intervals, transcribe words, classify intent, or calculate
risk.

Resource limits bound waveform length, frame count, batch size, and output memory.
Runtime problems are converted into stable project error codes instead of exposing
uncontrolled internal errors.

This task added 72 focused tests, repeat-safe setup and model-installation scripts,
license notices, and a real-model smoke test. The real smoke test loaded the exact
1,245,151-byte model, scored 34 frames twice, and matched the pinned reference
output.

### A4.2 — Sample-exact speech intervals

The third task added `src/audio_sentinel/speech_segments.py`. It turns the VAD's
many short scores into clean speech intervals on the original prepared recording.

This step is necessary because the preparation pipeline deliberately creates
overlapping audio windows. Overlap makes it less likely that an important sound is
cut at a boundary, but it also means the same speech can appear in more than one
window. Simply counting every positive frame would duplicate the same evidence.

The extraction service therefore:

1. validates the pinned VAD model before reading project audio;
2. opens the bounded preparation manifest;
3. requires current `acoustic_and_speech` consent;
4. requires 16 kHz mono prepared audio;
5. selects one complete prepared-window duration;
6. verifies every selected PCM-16 WAV and its fingerprint;
7. removes only the zero padding recorded in the manifest;
8. converts window-relative VAD frames to absolute clip samples;
9. joins qualifying frames across overlapping windows;
10. creates deterministic, non-overlapping speech intervals; and
11. checks the source again before returning evidence.

Exact sample positions are the source of truth. Human-readable seconds are derived
from those samples. This prevents small rounding differences from slowly moving
timestamps away from the original audio.

Frames that overlap or touch are joined. A larger silence gap is joined only when
the caller explicitly allows that gap. The final interval's score is the highest
supporting VAD score, not a sum. This prevents duplicate windows from making the
same speech look more certain than it is.

Every interval remembers all source windows and VAD frames that supported it. At
this stage, transcript fields remain empty and the handling decision is visibly
`not_transcribed`. The task does not invent words before the transcription model
runs.

The result is immutable and has a semantic evidence ID. Equivalent content receives
the same ID even if it is processed on a different date. A different source, model,
policy, interval, or contribution changes the ID.

This task added 37 focused tests for offsets, overlapping windows, thresholds, gap
rules, short intervals, padding, determinism, consent, source changes, malformed
files, resource limits, safe errors, and immutability. Its real-model smoke test
scored 83 VAD frames across three overlapping windows twice and reproduced the
exact structural interval `[0, 25600)` under an explicit zero-threshold test policy.
That zero threshold exists only to guarantee structural coverage with generated
non-speech audio; it is not the production policy.

### B4.2 — Verified offline English transcription

The fourth task added `src/audio_sentinel/transcription.py`, a local-only wrapper
around Faster-Whisper `tiny.en`.

The model is the official Systran CTranslate2 conversion at pinned repository
revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`. Its four required runtime files,
individual sizes and fingerprints, combined inventory fingerprint, installation
marker, package versions, CPU device, compute type, and English-only capability are
all verified before use.

Normal model loading does not access the network. The only network-dependent step
is the explicit one-time installation performed by the setup script.

The wrapper accepts one independent 16 kHz mono speech waveform and fixes the
decoding behavior:

- English language;
- transcription rather than translation;
- greedy one-beam decoding;
- temperature zero;
- no previous-text conditioning;
- no prompt or hotwords;
- no word-level timestamps; and
- no hidden internal VAD.

These fixed settings reduce hidden state and make repeated runs easier to explain.
The internal VAD is disabled because B4.1 and A4.2 already own the speech boundary.
Adding another unseen speech filter inside transcription would make missing text
harder to diagnose.

The wrapper copies the input before giving it to the model, fully consumes the
model's lazy output while still inside the error boundary, and validates text,
tokens, timestamps, probabilities, language metadata, and resource use.

For nonempty text, it calculates a `derived_score` by converting the model's token
log scores into a normalized 0-to-1 value, weighted by token count. This is useful
for consistent handling, but it is not a calibrated probability that every word is
correct. Empty or whitespace-only output creates no transcript candidate.

This wrapper does not accept or reject the text. It reports the candidate and lets
the orchestration layer apply the shared A4.1 policy.

This task added 39 focused tests, exact dependency pins, repeat-safe installation,
license notices, and a real offline smoke test. The smoke test loaded the verified
78 MB model and transcribed the same one-second silence twice, producing the same
empty result both times.

### A4.3 — Safe transcription orchestration

The fifth task added `src/audio_sentinel/speech_transcription.py`. This is the
gatekeeper that connects exact A4.2 speech intervals to B4.2 transcription and then
enforces the A4.1 reliability policy.

Before processing, the orchestrator confirms that:

- the transcription model is the verified pinned model;
- the A4.2 result is intact and still transcript-free;
- the public evidence matches the diagnostic window, frame, segment, timestamp,
  VAD-score, model, and hash information;
- the semantic evidence ID is correct;
- the prepared manifest is still the same manifest;
- consent is still active for speech processing; and
- every selected window is still the expected file with the expected bytes.

Each final speech interval is rebuilt from the prepared windows using exact sample
intersections. Every sample must be covered. Where two windows overlap, the samples
must agree exactly. If they disagree, the operation stops instead of arbitrarily
choosing one copy.

Each rebuilt interval is sent to the transcriber in a separate model call. Audio,
text, prompts, and decoder history are not carried from one interval into the next.
This prevents one segment from silently influencing another.

After inference, the manifest, windows, and consent state are checked again. If a
source changes while the model is running, the entire operation fails. It does not
return a mixture of old and new evidence.

Every nonempty text proposal remains in the audit evidence, regardless of its
handling outcome. A separate `downstream_transcripts` collection contains only
accepted text. Future language analysis is expected to use that collection rather
than reading every model guess.

The four possible outcomes are:

- `not_transcribed`: the model returned no usable text;
- `rejected_low_confidence`: the score was below the review boundary;
- `review_required`: the score was uncertain and automatic use is blocked; or
- `accepted`: the score met the acceptance boundary and may continue.

This design preserves evidence without allowing uncertain text to influence an
automatic decision. Missing text is not treated as proof of harmless speech.

The completed evidence adds the transcription model, candidates, assessments, and
counts while preserving the original source, VAD, window, segment, and policy
history. It contains no waveform or absolute filesystem path.

This task added 37 focused tests for all four outcomes, exact thresholds,
acceptance-only handoff, overlapping-window reconstruction, tampering, consent
expiry, deterministic identity, model failures, resource limits, privacy, and
immutability. The real orchestration smoke test rebuilt a 1.6-second interval from
three overlapping windows and ran the pinned transcriber twice. Both runs produced
the same evidence and correctly kept the generated no-text result out of the
downstream list.

### B4.3 — Model-wrapper boundary testing

The sixth task expanded the isolated tests around the two model boundaries. The
focused wrapper suite finished with 184 fast offline cases:

- 90 VAD cases; and
- 94 transcription cases.

These tests use synthetic waveforms and contract-accurate fake model runtimes.
They do not need the large optional runtimes or downloaded model files, so they can
run quickly as part of ordinary development.

The VAD matrix covers successful loading and inference plus unsafe paths, links,
junctions, missing and extra files, wrong fingerprints, changes during loading,
runtime and provider drift, changed tensor contracts, exact size and memory limits,
frame construction, preceding context, recurrent state, batching, final padding,
malformed probabilities, safe failures, immutable results, and portable metadata.

The transcription matrix covers the complete model inventory, runtime versions,
CPU and English capability, local-only loading, fixed decoding options, input
ownership, normalization, confidence calculation, empty output, malformed model
segments, decreasing timestamps, token and UTF-8 byte limits, lazy-iterator
failures, memory failures, safe errors, immutable results, and portable metadata.

The purpose of this matrix is to make both wrappers fail closed. In common terms,
if the model files, runtime, input, output, or limits are not exactly what the code
expects, the wrapper stops instead of producing believable-looking evidence from
an unverified state.

### A4.4 — Complete branch integration testing

The seventh task added `tests/test_speech_integration.py` with 16 integration
cases. These tests connect the real preparation, persistence, manifest checking,
VAD orchestration, segment merging, waveform reconstruction, transcription
orchestration, reliability policy, and evidence-building code.

Only the two expensive pretrained runtime calls are replaced with
contract-accurate fakes. This gives the test precise control over model answers
while still exercising the real application code around them.

The integration matrix proves the following connected behaviors:

- a 48 kHz stereo recording becomes 16 kHz mono prepared audio;
- overlapping support becomes one speech interval and one transcription call;
- WAV and FLAC inputs at 8, 16, 44.1, and 48 kHz work across mono and stereo cases;
- rejected, review-required, accepted, and empty transcripts keep their order;
- only accepted text reaches the downstream collection;
- no-speech input skips the transcription model;
- final-window zero padding never reaches VAD evidence or transcription;
- a custom policy is preserved across both model stages;
- fresh services and model objects reproduce the same semantic identities;
- processing can continue from the verified prepared bundle after raw input removal;
- successful processing does not modify the raw source or prepared files;
- a VAD failure stops before transcription;
- a later transcription failure returns no partial final evidence;
- a window changed between stages is rejected before transcription; and
- final evidence survives a JSON round trip without waveforms or absolute paths.

Running the connected suite exposed an order-dependent defect in an earlier VAD
test. That test reloaded the production module inside the shared pytest process,
which replaced Python class identities still referenced by tests that ran later.
The check was moved into a child Python process. It still proves that importing the
normal package does not load ONNX Runtime, but it no longer alters shared test
state. This was a test-isolation correction, not a production behavior change.

## 4. Why the audio timing is sample-exact

Audio time is ultimately a sequence of samples. At 16 kHz, one second contains
16,000 samples. Using sample positions as the primary coordinates avoids ambiguity.

For example:

```text
start sample: 8,000
end sample: 24,000
sample rate: 16,000 samples/second

start time = 8,000 / 16,000 = 0.5 seconds
end time   = 24,000 / 16,000 = 1.5 seconds
```

The code stores the exact sample bounds and derives seconds from them. It never
repeatedly converts rounded seconds back into samples. This matters when windows
overlap, intervals touch, or the last window contains padding.

## 5. How overlapping windows are handled

Phase 1 creates overlapping windows so speech near a window edge also appears near
the middle of a neighboring window. That improves coverage but creates duplicate
observations.

Phase 4 handles this in two places:

1. VAD frames are converted from window-relative locations into one absolute clip
   timeline. Overlapping or adjacent positive support is unioned into one interval.
2. When the interval is rebuilt, overlapping windows must contain exactly the same
   audio samples. The shared samples are used once, not repeated.

The strongest supporting VAD score is retained. Scores are not added together,
because two views of the same speech should not look like two independent votes.

## 6. Reliability handling in everyday terms

The policy acts like a four-lane checkpoint:

| Lane | What happened | What the system does |
| --- | --- | --- |
| No result | The model produced no usable words. | Records `not_transcribed` and sends nothing forward. |
| Reject | The text score is below 0.50. | Keeps the proposal for audit but blocks it. |
| Review | The score is at least 0.50 but below 0.80. | Marks it for a future human-review process and blocks automation. |
| Accept | The score is at least 0.80. | Keeps the evidence and copies the text into the downstream list. |

The evidence record and downstream list serve different purposes. The evidence
record answers, "What did the model produce?" The downstream list answers, "What
is currently permitted to influence the next automatic stage?"

This separation is one of the most important safeguards in the phase.

## 7. Security, privacy, and audit safeguards

### Local processing

Both speech models run locally after one explicit setup. Normal loading and
inference do not upload recordings or fetch replacement models from the internet.

### Current consent

Speech processing requires active `acoustic_and_speech` permission. Consent is
checked at important boundaries, including before and after model work. Expired or
changed permission invalidates the operation.

### Pinned models and runtimes

Model files, versions, fingerprints, runtime packages, execution provider, tensor
contracts, decoding settings, and capabilities are pinned and checked. This makes
the result traceable to one known configuration.

### Safe filesystem boundaries

Managed model and project files must remain in approved directories. Traversal,
links, junctions, unexpected files, and path substitutions are rejected where they
could change the meaning of the input.

### Change detection

Raw audio, manifests, prepared windows, and model artifacts have digital
fingerprints. Important sources are checked again after processing so a file cannot
be changed mid-run without detection.

### Bounded resource use

Limits cover file bytes, decoded samples, frame counts, batch sizes, model output
memory, speech-segment counts, total transcription samples, token counts, text
bytes, and evidence size. Oversized work fails predictably.

### No partial success

If a later segment fails transcription or a source changes during processing, the
orchestrator does not return a partly completed final result. The whole operation
fails with a stable error.

### Minimal portable evidence

Final evidence contains the information needed to explain the result but excludes
raw waveforms, absolute local paths, speaker embeddings, and speaker identity.
Transcript text remains sensitive consent-controlled evidence.

### Immutable outputs

Returned contracts and metadata cannot be silently modified in place. A caller must
create a new valid result rather than changing an old receipt behind the system's
back.

## 8. Determinism and reproducibility

Phase 4 uses fixed model versions and decoding options, deterministic ordering,
sample-based timing, recorded policies, and semantic evidence IDs.

A semantic ID is based on meaningful content rather than the clock time. If the
same verified source, model, policy, segment, and transcript content is processed
again, the semantic identity stays the same even though the new receipt has a
different creation time.

This supports three practical questions:

- Did the same input and rules produce the same meaningful result?
- Did any source, model, policy, or output change?
- Can an auditor reconstruct why text was accepted or blocked?

The system does not promise that all hardware and every future model runtime will
always produce identical floating-point values. It verifies the specific pinned
local configuration used by this project.

## 9. Testing and verification completed

Phase 3 ended with 717 passing project tests. Phase 4 ended with 1,034, a net
increase of 317 tests.

| Work area | Tests at completion | Purpose |
| --- | ---: | --- |
| A4.1 speech contract | 43 added | Schema, policy, validation, identity, and privacy |
| B4.1 VAD wrapper | 72 added | Verified model boundary and frame scoring |
| A4.2 speech segments | 37 added | Exact timeline placement and overlap merging |
| B4.2 transcription wrapper | 39 added | Verified local decoding and candidate creation |
| A4.3 orchestration | 37 added | Reconstruction, policy handling, and final evidence |
| B4.3 wrapper hardening | 184 total wrapper cases | Expanded VAD and transcription boundary coverage |
| A4.4 integration | 16 added | Complete cross-component speech branch |

The B4.3 total includes earlier B4.1 and B4.2 wrapper tests. It increased those
existing files rather than creating 184 additional tests. Counting only new Phase 4
tests across the repository gives the 317-test net increase.

The final verification performed for A4.4 included:

- **16 passing integration cases** for the connected branch;
- **274 passing speech cases** across VAD, segmentation, transcription, orchestration,
  and integration;
- **1,034 passing project tests** in the full verification script;
- a successful real Silero model load and repeat check;
- a successful real speech-segment extraction smoke test;
- a successful real Faster-Whisper model load and repeat check;
- a successful real end-to-end structural orchestration smoke test; and
- a clean project preparation smoke check.

The ordinary unit and integration suites remain offline. Contract-accurate fake
runtimes make model answers predictable, which lets the tests precisely cover
successes, failures, thresholds, and tampering. Separate smoke tests prove that the
real pinned artifacts and runtimes can be loaded and connected.

## 10. Model setup and manual steps

On a fresh machine, run:

```powershell
.\scripts\setup_speech.ps1
```

That repeat-safe setup performs the following work:

1. creates or reuses `.venv/speech`;
2. installs the exact speech runtime dependencies;
3. obtains the pinned Silero and Faster-Whisper model artifacts;
4. verifies downloaded files before publishing them locally;
5. keeps the large model files in Git-ignored directories; and
6. runs the real VAD, segmentation, transcription, and orchestration smoke tests.

The model download is the network-dependent part. Normal speech processing after
setup is local and offline.

The current development machine has already completed this setup successfully. No
manual action was required for the normal 1,034-test project suite.

## 11. Important limitations and non-goals

Phase 4 is complete as an engineering baseline, but it does not establish that the
speech models are accurate enough for a real safety deployment.

The following limitations remain:

- The tests verify software behavior and model contracts, not real-world VAD or
  transcription accuracy.
- There is no labeled speech benchmark yet covering accents, dialects, age groups,
  background noise, distance, rooms, microphones, overlapping speakers, or device
  conditions expected in deployment.
- Faster-Whisper `tiny.en` is English-only. Other languages are outside this phase.
- A derived transcript score is not a calibrated probability that the words are
  correct.
- The default 0.50 and 0.80 transcript boundaries are policy baselines, not
  deployment-approved operating points.
- `review_required` records a need for review, but a human-review product workflow
  has not yet been implemented.
- No speaker is identified, verified, or enrolled.
- The system does not determine who said the words.
- The system does not classify threats, distress, harmless negation, or intent.
- The system does not create an incident, assign severity, calculate risk, or send
  an alert.
- No text result does not mean that no speech occurred, and it never proves that
  the recording was safe.
- Accepted text may still contain recognition errors and must be treated as one
  evidence source rather than unquestionable fact.
- Real deployment still needs performance, latency, privacy, hardware, and
  environment-specific evaluation.

## 12. What Phase 4 hands to Phase 5

Phase 5 should consume only the `downstream_transcripts` collection produced by
A4.3. Each item contains accepted text plus its exact segment time and typed
confidence information.

Phase 5 must not reinterpret rejected or review-required text as accepted. It also
must not interpret a missing transcript as harmless language.

The next task, A5.1, will define:

- the allowed language-analysis categories;
- the reason codes explaining each category;
- the evidence schema for language findings; and
- the boundary between text evidence and later risk decisions.

In common terms, Phase 4 answers, "What sufficiently reliable words might have been
spoken, and where?" Phase 5 will begin answering, "What limited, explainable
language category do those accepted words support?"

## 13. Main files produced during the phase

### Production modules

- `src/audio_sentinel/speech_contracts.py` — evidence types and reliability policy
- `src/audio_sentinel/vad.py` — verified Silero loader and frame scoring
- `src/audio_sentinel/speech_segments.py` — exact speech-interval extraction
- `src/audio_sentinel/transcription.py` — verified Faster-Whisper wrapper
- `src/audio_sentinel/speech_transcription.py` — reconstruction and policy gate

### Tests

- `tests/test_speech_contracts.py`
- `tests/test_vad.py`
- `tests/test_speech_segments.py`
- `tests/test_transcription.py`
- `tests/test_speech_transcription.py`
- `tests/test_speech_integration.py`

### Setup and real-model checks

- `scripts/setup_speech.ps1`
- `scripts/install_silero_vad.py`
- `scripts/install_whisper_tiny_en.py`
- `scripts/smoke_test_vad.py`
- `scripts/smoke_test_speech_segments.py`
- `scripts/smoke_test_transcription.py`
- `scripts/smoke_test_speech_transcription.py`

### Contracts, examples, and supporting documentation

- `docs/schemas/v1/speech-evidence.schema.json`
- `docs/examples/speech-evidence.json`
- `docs/speech-evidence.md`
- `docs/vad-wrapper.md`
- `docs/speech-segments.md`
- `docs/transcription-wrapper.md`
- `docs/speech-transcription.md`
- `docs/speech-wrapper-tests.md`
- `docs/speech-integration-tests.md`

## 14. Short glossary

**VAD:** Voice activity detection. A model estimates whether a short audio piece
sounds like speech; it does not understand the words.

**Frame:** A very short piece of audio examined by the VAD model.

**Window:** A larger saved piece of the prepared recording. Windows overlap to
avoid losing sounds at their edges.

**Speech segment:** A clean interval on the full recording timeline supported by
one or more qualifying VAD frames.

**Transcript candidate:** Text proposed by the transcription model before the
reliability policy decides how it may be used.

**Derived score:** A normalized value calculated from the model's token scores. It
helps apply consistent rules but is not a proven probability of correctness.

**Provenance:** The recorded history of which source, model, runtime, settings, and
rules produced an output.

**SHA-256:** A digital fingerprint used to detect changed file contents.

**Semantic evidence ID:** A fingerprint of the meaningful evidence content. It
does not change merely because the same work was run at a different time.

**Fail closed:** Stop and return no trusted result when verification fails, instead
of guessing or continuing with questionable data.

## 15. Final status

Phase 4 is complete. Project 1 now has a local, consent-controlled, traceable speech
branch that can move from verified prepared audio to sample-exact speech intervals,
offline English transcript candidates, reliability decisions, and an
acceptance-only downstream handoff.

The phase does not claim to understand intent or prove an emergency. Its job is to
produce controlled speech evidence for Phase 5 while keeping uncertain results,
changed inputs, broken models, and partial failures from silently influencing later
automation.

Tracker status after Phase 4: **37 completed, 0 in progress, and 35 not started out
of 72 total tasks.**

Next task: **A5.1 — Define language categories, reason codes, and evidence schema.**
