# B1.3: Deterministic overlapping window segmentation

`src/audio_sentinel/segmentation.py` now divides a `PreparedSignal` into fixed-length
audio windows. It uses the A1.1 settings and produces `PreparedWindowRecord` metadata
accepted by the existing manifest validator. It does not write audio files; that
is A1.3. No installation, recording, or dataset download is needed for this task.

## Window rules

Default durations are 1, 5, and 10 seconds, with 50% overlap. Each duration is a
separate view of the same full clip. For each duration, window length is
`round(seconds * sample_rate_hz)` and hop is `round(length * (1 - overlap))`.
Rounding uses ties to even, as defined in A1.1. Start positions are computed from
integer sample indices, avoiding accumulation of floating-point timing errors.

With `pad`, stop at the first window that reaches the clip end. If necessary, append
zeros to that final window. Clips shorter than the duration produce one padded
window. An exact fit produces no extra tail; no all-padding window is created.
With `drop`, incomplete windows are omitted; short clips can produce zero windows.

A 1.6-second clip at 16 kHz has 25,600 sample frames. Its default output is:

| Duration group | Start frame | End frame, exclusive | Added zero frames |
| --- | --- | --- | --- |
| 1 second | 0 | 16,000 | 0 |
| 1 second | 8,000 | 24,000 | 0 |
| 1 second | 16,000 | 25,600 | 6,400 |
| 5 seconds | 0 | 25,600 | 54,400 |
| 10 seconds | 0 | 25,600 | 134,400 |

Padding belongs only to the window array. It never extends the original clip's
duration, annotations, or `end_sample`. Stereo uses the same frame positions in
both channels. Seconds for a boundary are `sample_index / sample_rate_hz`.

## API and memory use

```python
from audio_sentinel.segmentation import iter_windows

# prepared = prepare_signal(loaded, audio_settings)
for window in iter_windows(prepared):
    print(window.record.start_sample, window.record.end_sample)
    print(window.samples.shape, window.record.padding_samples)
```

Each `AudioWindow` contains the source clip ID, prepared sample rate, an independent
float32 `(frames, channels)` array, and a `PreparedWindowRecord`. Windows arrive in
duration order, then start-position order. Both padded and full windows own their
storage, so editing one does not change another window or the source signal.

The function validates input and creates a small plan eagerly, then returns an
iterator that allocates one window at a time. Consume and release windows as you
go. Converting the iterator to a list keeps every overlapping array in memory and
can be much larger than the original clip. The source signal remains in memory;
do not modify it while iterating.

The source and each emitted window must fit `max_decoded_bytes`. This includes
padding: a small source with a very long window can still exceed the limit. That
failure is reported before iteration. Windows omitted by `drop` need no allocation.
The limit is per array, not a total process-memory limit.

## Metadata and deterministic names

The record contains window ID, planned relative audio path, requested duration,
start/end frame positions, and padding count. A SHA-256 key over the clip ID,
original source hash, complete settings, and preprocessing version creates an
opaque output namespace. Each name also includes its duration-group index and
integer start position. Consequently:

- Identical input metadata and settings produce identical records and names.
- Different duration groups remain distinct even when their rounded lengths match.
- Changes to source hash, clip ID, or settings select a different namespace.
- Long identifiers and identifiers containing colons do not become unsafe filenames.

Paths follow `prepared/<key>/windows/g0000-s000000000000.wav`, relative to
`data/interim/`. They are planned destinations only; no such file exists until
persistence writes it. The key identifies the source/settings combination, not a
checksum of the future encoded WAV. A1.3 can persist these records without
recalculating boundaries. Tests validate complete generated inventories against
`PreparedAudioManifest` for both padding policies.

## Validation and permission

The segmenter rejects invalid/non-finite samples, invalid clip IDs, mismatched
prepared rate or dimensions, and array memory limits. Errors use the existing
`AudioTransformError` codes. Permission is validated on entry and again before each
window is generated, so a paused iterator cannot continue past its consent snapshot's
expiry. Permission errors remain `AudioLoadError`. Tests can supply an aware `now=`
clock value; production callers should use the real clock by omitting it.

No transcription, relabeling, noise reduction, or per-window normalization happens
here. Each window contains samples from the already prepared full clip plus any
required trailing zeros.

## Run the synthetic check

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_segmentation.py
```

This generates a temporary 1.6-second stereo tone at 48 kHz, loads it, prepares it
as 16 kHz mono, and verifies the five windows shown above. The temporary source is
removed afterward. Run `./scripts/verify_project.ps1` for the full test suite.

## In plain language

Windows are short excerpts that later models can analyze. Overlap lets a sound near
the edge of one excerpt appear more centrally in another. For 1-second windows with
50% overlap, the next excerpt starts every half-second. If the final excerpt is too
short, zero padding fills its remaining space while the metadata records which
part came from the real recording.
