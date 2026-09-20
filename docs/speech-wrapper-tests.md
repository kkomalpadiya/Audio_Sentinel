# B4.3 — VAD and transcription wrapper tests

## What this task verifies

B4.3 completes the focused regression matrix around the two isolated model
boundaries introduced by B4.1 and B4.2. The tests use synthetic waveforms, fake
ONNX sessions, and fake Faster-Whisper outputs. The ordinary suite therefore stays
fast, offline, deterministic, and independent of the optional speech runtimes and
downloaded model files.

The focused suite contains 184 cases: 90 for `test_vad.py` and 94 for
`test_transcription.py`.

## VAD coverage

The VAD tests verify successful local CPU loading, portable model metadata, exact
frame-grid and tail-padding calculations, preceding-frame context, recurrent-state
carry within one waveform, state reset between waveforms, and deterministic batch
splitting.

Failure and boundary coverage includes:

- missing, traversing, absolute, linked, or junction-backed model paths;
- exact artifact inventory, marker, file-size, byte-limit, digest, and mid-read or
  mid-load change detection;
- missing or wrong ONNX Runtime versions, runtime-construction failures, provider
  drift, uninspectable sessions, and complete input/output signature drift;
- one-dimensional finite float32 input, valid amplitude endpoints, exact input and
  output limits, model-input ownership, and preparation memory failure;
- every pinned metadata field and the callable session entry point;
- probability and recurrent-state dtype, shape, finiteness, inventory, and range;
  and
- immutable JSON-ready results without waveform or absolute-path disclosure.

## Transcription coverage

The transcription tests verify the complete pinned four-file inventory, aggregate
identity, exact runtime versions, CPU-only English model capability, local-only
loading, the complete fixed decoding option set, owned contiguous model input,
text normalization, token-weighted confidence calculation, and empty-output
handling.

Failure and boundary coverage includes:

- unsafe model paths, linked files, malformed or oversized markers, missing or
  extra files, wrong payload sizes or hashes, and changes during hashing or loading;
- missing or drifted Faster-Whisper, CTranslate2, and tokenizer runtimes, model-load
  failures, capability drift, and a missing transcription entry point;
- waveform type, rank, dtype, finiteness, range, and exact input-limit behavior;
- segment IDs, ordering, timestamps, text/control characters, tokens, log
  probabilities, no-speech probabilities, English-only metadata, and lazy iterator
  failures;
- exact segment, token, and UTF-8 output-byte boundaries, including resource use by
  whitespace-only hypotheses;
- every pinned metadata field, model-call failures, distinct memory failures, and
  safe error messages; and
- immutable, JSON-ready results without retained waveforms.

## Commands

Run the fast focused suite:

```powershell
python -m pytest -q tests/test_vad.py tests/test_transcription.py
```

Run the real pinned-model checks separately:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_vad.py
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_transcription.py
```

The real smoke checks validate the locally installed model artifacts and runtimes.
They do not measure speech-detection or transcription accuracy; these tests verify
the software boundary, not model quality.

A4.4 builds on these isolated checks with the connected integration matrix in
[`speech-integration-tests.md`](speech-integration-tests.md).

In plain language: both speech-model adapters now have explicit tests for their
normal behavior and for the dangerous cases where files, runtime contracts, model
outputs, or limits are wrong. The wrappers must fail closed instead of returning
plausible-looking evidence from an unverified state.
