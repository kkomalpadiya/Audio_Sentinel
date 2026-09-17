# A3.2 — YAMNet inference over prepared waveforms

## What this task adds

`audio_sentinel.acoustic_inference.infer_prepared_audio` sends every saved
preparation window through the verified local YAMNet v1 callable in manifest
order. It consumes the prepared 16 kHz mono PCM16 WAVs, not the Phase 2 generic
Log-Mel arrays. YAMNet owns the frontend it was trained with.

The result retains each raw `(patches, 521)` float32 sigmoid-score matrix. It also
records the model metadata, manifest/raw/window hashes, exact source window,
unpadded model-input length, embedding shape, model spectrogram shape, and the
absolute sample support of each patch. The score arrays are owned, contiguous,
read-only copies; embeddings and spectrogram values are validated and discarded
because later project stages need the class scores, not those large intermediates.

This task does **not** select thresholds, combine mapped classes, merge overlapping
windows, create events, estimate risk, or claim that a score is an incident
probability. B3.2 owns temporal aggregation and A3.4 owns measured threshold and
accuracy decisions.

## Padding and patch timing

Preparation windows may contain zero padding at the end so their WAV files keep a
fixed duration. Inference verifies that padding is zero and removes it before the
model call. YAMNet then applies its own documented right padding to form complete
0.975-second patches with 0.48-second hops. Reported patch support is clipped to
the real prepared-audio span; padded time is never presented as recorded evidence.

For example, a one-second window produces two model patches. Their raw support is
`0–15,600` and `7,680–23,280` input samples, but the second span is reported only
through sample `16,000` because the remaining model context is padding.

## Verification and limits

Before inference, the service requires the exact loaded model/vocabulary hashes
from B3.1, a 16 kHz mono preparation manifest, active acoustic-processing consent,
and a bounded number of files. Each WAV must match its manifest record exactly.
Decoded input and predicted output memory are bounded before the model call.

After inference, the service checks all three model outputs for exact shape,
float32 dtype, finite values, and score range. It rechecks the manifest and every
window hash so it cannot return a result assembled from source files that changed
during the run. Errors expose stable codes without leaking TensorFlow details.

Default limits are 16 MiB for the manifest, 256 MiB for one window and its decoded
float32 samples, 512 MiB across model outputs, and 10,000 windows. Override these
with `AcousticInferenceSettings` only after considering available memory.

## Running it

The one-time YAMNet setup from B3.1 is still required on a fresh checkout:

```powershell
.\scripts\setup_yamnet.ps1
```

Run the real generated-tone inference smoke test with the isolated interpreter:

```powershell
.\.venv\yamnet\Scripts\python.exe .\scripts\smoke_test_acoustic_inference.py
```

The ordinary unit suite uses a deterministic fake callable, so normal development
does not import TensorFlow or need model files.

In plain language: this step safely feeds each prepared excerpt into the sound
recognizer and keeps the recognizer's original class scores with proof of which
audio and model produced them. The next task will combine repeated evidence from
overlapping excerpts without counting the same sound twice.
