# Offline transcription wrapper (B4.2)

## Outcome

`audio_sentinel.transcription` provides a narrow, local-only boundary around the
English Faster-Whisper `tiny.en` model. It accepts one independent 16 kHz mono
`float32` speech segment and returns either a transcript candidate with diagnostic
chunks or an explicit empty result. It does not extract speech, apply the A4.1
reliability policy, classify language intent, create an incident, or calculate
risk. A4.3 owns orchestration and low-confidence handling.

The implementation uses the official
[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) CTranslate2 runtime and
the official [Systran tiny.en conversion](https://huggingface.co/Systran/faster-whisper-tiny.en).
The repository revision, four runtime files, individual byte sizes and SHA-256
digests, combined inventory digest, package versions, CPU device, and resolved
compute type are pinned.

## Trust boundary

`load_transcription_model` accepts only
`models/faster-whisper-tiny.en/1/`. The directory must contain exactly:

- `config.json`
- `model.bin`
- `tokenizer.json`
- `vocabulary.txt`
- `.complete.json`

Links, junctions, traversal, missing or additional files, size changes, digest
changes, marker changes, runtime-version drift, and changes while loading are
rejected. `local_files_only=True` prevents the runtime from fetching a replacement
model. Loading is CPU-only, single-worker, single-threaded, and must resolve to
CTranslate2 `int8_float32`. The English-only capability is inspected before the
model is returned.

## Deterministic decoding

`transcribe_segment` fixes the language to English and the task to transcription.
It uses greedy one-beam decoding at temperature zero, disables previous-text
conditioning, word timestamps, and the internal VAD, and supplies no prompt or
hotwords. B4.1/A4.2 already own speech gating, so a second hidden VAD would make
the pipeline harder to reproduce.

The wrapper validates the input before inference and passes an owned contiguous
copy to the runtime. It exhausts Faster-Whisper's lazy segment iterator inside the
error boundary, validates every timestamp, token, score, and language field, and
enforces configurable input, segment, token, and output-byte limits. Model errors
are converted to stable safe error codes.

## Confidence meaning

Faster-Whisper exposes an average token log probability for each returned chunk,
not a calibrated probability that the words are correct. The wrapper therefore
computes a `derived_score`: the exponential of the token-count-weighted mean of
the chunk average log probabilities, clamped to `[0, 1]`. Empty text returns no
candidate. The score is labeled `derived_score`; only A4.3 may pass it through the
recorded A4.1 review and acceptance thresholds.

`language="en"` and `language_confidence=1.0` record the model's fixed English-only
capability. They are not a measured language-detection probability.

## Setup and checks

On a fresh machine:

```powershell
.\scripts\setup_speech.ps1
```

The setup creates or reuses `.venv/speech`, installs exact transcription runtime
versions, downloads only the four files at the pinned Hugging Face revision,
verifies them in a staging directory, atomically publishes the ignored local model,
and runs all speech smoke checks. The download is the only network-dependent step;
normal loading and inference are offline.

To rerun just this real-model check:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_transcription.py
```

The smoke test loads the real verified model and transcribes the same one-second
silence twice. Both runs must yield the same empty candidate. Positive transcript,
normalization, confidence, lazy-generator, failure, and limit behavior are covered
with isolated unit tests so the normal test suite remains model-download-free.

B4.3 completes the 94-case transcription matrix documented in
[`speech-wrapper-tests.md`](speech-wrapper-tests.md). It adds full artifact,
runtime, metadata, malformed-output, exact-limit, input-ownership, and safe-failure
coverage around this wrapper.

The Faster-Whisper runtime and original OpenAI Whisper model are MIT licensed; the
notices are retained under `src/audio_sentinel/resources/`.
