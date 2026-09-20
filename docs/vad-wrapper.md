# Silero VAD v6 wrapper

## What B4.1 adds

B4.1 implements the pretrained voice-activity-detection boundary in
`src/audio_sentinel/vad.py`. It installs and verifies a local Silero VAD v6 ONNX
artifact, loads it with a pinned CPU-only ONNX Runtime, and returns one raw speech
probability for every 512 samples of an independent 16 kHz waveform.

This task does not merge probabilities into speech segments, apply the A4.1
reliability threshold, read preparation manifests, or transcribe audio. A4.2 owns
segment extraction and absolute timestamp handling.

## Selected model and provenance

The wrapper uses Silero VAD v6 from the asset distributed in
`faster-whisper==1.2.1`. That release upgraded Faster Whisper to Silero VAD v6, and
the package includes the ONNX asset needed by the wrapper:

- Silero VAD release: <https://github.com/snakers4/silero-vad/releases/tag/v6.0>
- Faster Whisper release: <https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.2.1>
- Pinned package wheel SHA-256:
  `79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7`
- Pinned ONNX asset SHA-256:
  `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2`
- Pinned ONNX asset size: 1,245,151 bytes
- Runtime: `onnxruntime==1.23.2`, CPU execution provider only

Silero VAD and Faster Whisper use the MIT license. Their license notices are checked
in under `src/audio_sentinel/resources/`.

## Installation

Run once on a fresh machine:

```powershell
.\scripts\setup_speech.ps1
```

The script creates the ignored `.venv/speech/` environment, installs the pinned
`speech` dependency extra, verifies the packaged model bytes, copies the model into
ignored `models/silero-vad/6/`, writes a strict installation marker, and runs the
real-model smoke test. Re-running the command verifies and reuses a valid model.
It refuses to overwrite an unexpected existing model directory.

Normal package imports do not import ONNX Runtime. The runtime is loaded only when
`load_silero_vad()` is called.

## Verified model contract

The loader rejects the model before use unless all of the following match:

- exact local file inventory and installation marker;
- model byte size and SHA-256 digest;
- regular files inside the approved project model directory, with no links or
  junctions;
- `onnxruntime` version 1.23.2 and CPU-only execution;
- inputs `input [sequence, 576]`, `h [1, 1, 128]`, and `c [1, 1, 128]` as float32;
  and
- outputs `speech_probs [sequence]`, `hn [1, 1, 128]`, and `cn [1, 1, 128]` as
  float32.

The loader hashes the model both before and after ONNX Runtime opens it. A file that
changes during either verification pass is rejected.

## Frame scoring behavior

`infer_vad_probabilities()` accepts one finite, one-dimensional float32 waveform
whose values remain within `[-1, 1]`. The prepared-audio target sample rate is
16 kHz, so each 512-sample frame represents 32 milliseconds.

For every frame the wrapper supplies 64 samples of preceding audio context. The
first frame uses zero context. The wrapper right-pads only the final incomplete
frame and records how many padded samples were used. Reported end offsets and times
stop at the last real sample, so later code cannot mistake padding for recorded
audio.

Silero's recurrent `h` and `c` states are carried across bounded batches within one
waveform. Both states reset to zero for every independent call. This prevents one
prepared window from changing the result of a later, unrelated window.

The result contains:

- verified model and runtime metadata;
- real and padded sample counts;
- frame index and sample-exact start/end offsets;
- relative start/end seconds;
- the raw speech probability; and
- per-frame tail padding.

Probabilities must be finite float32 values in `[0, 1]`. They remain raw model
evidence. B4.1 does not label a frame as speech and does not claim the score is a
calibrated probability of correctness.

## Resource and failure controls

The default wrapper limits one call to 9,600,000 samples (10 minutes at 16 kHz),
10,000 frames per ONNX invocation, and 16 MiB of returned probability data. Invalid
waveforms, oversized inputs, output-shape drift, NaN/Infinity, out-of-range scores,
runtime failures, and model mismatches produce stable `VadError` codes without
exposing private runtime details.

## Verification

Run the TensorFlow- and ONNX-free unit suite with the ordinary project environment:

```powershell
python -m pytest tests/test_vad.py -q
```

B4.3 expands this into the complete 90-case VAD matrix documented in
[`speech-wrapper-tests.md`](speech-wrapper-tests.md), including mid-load changes,
exact resource boundaries, full metadata drift, memory failure, and input ownership.

Run the real pinned model check with:

```powershell
.\.venv\speech\Scripts\python.exe .\scripts\smoke_test_vad.py
```

The smoke test verifies the model and runtime, scores 34 frames, repeats the call to
check state reset, and proves the wrapper output is array-exact with the pinned
Faster Whisper Silero implementation.

## Downstream use

A4.2 reads verified prepared windows, applies the recorded VAD reliability policy,
merges qualifying frames into non-overlapping speech segments, and converts window-
relative spans into sample-exact prepared-clip timestamps. A4.3 then reconstructs
those segments for transcription.

## Beginner-friendly explanation

The VAD wrapper is a careful adapter around a pretrained speech detector. It chops
audio into 32-millisecond pieces and asks the model how speech-like each piece is.
It also checks that the exact approved model is being used and resets the model's
memory between unrelated clips. The next task will turn these small scores into
larger speech intervals.
