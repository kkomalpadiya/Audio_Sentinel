# A1.2: Mono conversion, resampling, and volume normalization

A1.2 now converts the numeric audio returned by B1.1 into a consistent signal in
memory. The implementation is `src/audio_sentinel/audio_transforms.py`.
No datasets or dependencies need to be downloaded on the current project machine.

## The three operations

**Mono conversion:** `convert_to_mono(samples)` averages the channels at each moment
and returns an array shaped `(frames, 1)`. Accumulating in float64 avoids overflow
when averaging large finite float32 inputs. Mono input also returns an independent
copy. Averaging can cancel opposite-phase channels; it follows the A1.1 policy and
does not choose a microphone or try to repair phase differences. Set
`convert_to_mono=False` to preserve separate channels.

**Resampling:** `resample_audio(samples, source_rate_hz, target_rate_hz,
max_decoded_bytes=...)` changes the number of samples per second along the time
axis. The default target is 16,000 Hz. It uses
[SciPy's polyphase FIR resampler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html),
with a Kaiser window of 5.0 and zero extension outside the clip. The filter reduces
frequencies above the new usable range before downsampling. This avoids turning
high-frequency content into false lower-frequency content, called aliasing.

The result has exactly `round(source_frames * target_rate / source_rate)` frames,
using Python's ties-to-even rounding. SciPy's result is trimmed or padded at the end
to match this contract. This is separate from later window padding. A zero-frame
result is rejected. Same-rate input bypasses filtering and returns an exact copy.
Clip duration stays the same within half a target sample due to rounding; pitch
and playback speed are preserved. Filtering can affect the very beginning and end.

**Volume normalization:** `normalize_loudness(samples, settings)` measures RMS over
the whole signal, across all channels. RMS summarizes average signal amplitude.
The target is −20 dBFS by default. A single multiplier adjusts every sample,
preserving relative changes over time and the balance between channels.
It uses the smallest of the gain needed for the RMS target, the +20 dB amplification
limit, and the gain allowed by the −1 dBFS peak ceiling. Limits take priority over
reaching the target; the result can therefore remain quieter than −20 dBFS.

Disabled normalization and clips at or below the −60 dBFS RMS silence floor are
copied unchanged. That bypass also skips peak limiting. The floor comparison uses
the nearest float32 amplitude so an exactly represented floor value is not boosted.
For normalized clips, the peak ceiling is rounded down to a representable float32
value. A single gain enforces the ceiling without flattening individual peaks.
There is no PCM encoding or hard clipping in this task.

`NormalizedAudio.stats` reports the applied gain, RMS and peak before/after, and a
reason: `target`, `gain_limit`, `peak_limit`, `silence`, or `disabled`. Zero amplitude
has no finite dBFS value, so its RMS/peak fields are `None`, which becomes JSON null.
This is RMS normalization, not perceptual LUFS normalization.

## Using loader output

```python
from audio_sentinel.audio_transforms import prepare_signal

# loaded = load_audio(clip, project_settings), as described in audio-loading.md
prepared = prepare_signal(loaded, project_settings.audio)
print(prepared.samples.shape)
print(prepared.sample_rate_hz, prepared.duration_seconds)
print(prepared.normalization)
```

`prepare_signal` averages channels when enabled, resamples, and then normalizes.
It returns `PreparedSignal` with the samples, target rate, settings snapshot,
normalization measurements, clip ID, consent, and a copy of the original source
metadata. Source metadata continues to describe the original recording; properties
such as `prepared.num_frames` describe the transformed signal.

The helper rechecks permission before and after work and verifies the sample shape
against source metadata. It rejects invalid/non-finite inputs and results, and
checks the configured main-array memory limit before resampling. This limit is not
a total process memory cap: source, result, intermediate arrays, and SciPy's filter
workspace can coexist. Transform functions do not mutate their input arrays.

`AudioTransformError` provides a stable `code` for invalid samples/rates, non-finite
audio, memory limits, zero-length output, source-shape mismatch, or unavailable
noise reduction. Permission errors remain `AudioLoadError`, as in B1.1. Standalone
numeric functions accept float32 arrays; permission handling is at the loaded-clip
boundary. Extremely large finite values that overflow the resampling filter are
rejected if they produce non-finite output.

## Relationship to the remaining tasks

Noise reduction remains disabled by default. If explicitly enabled, this helper
raises `noise_reduction_unavailable` so it cannot silently claim to have applied it.
B1.2 will add that step between resampling and normalization. Segmentation, writing
WAV files/manifests, and full preparation-service integration remain B1.3, A1.3, and
A1.4. `PreparedSignal` is an in-memory result, so it does not pretend to be the
existing file-backed `PreprocessedAudio` interface yet.

## Try the synthetic example

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_test_audio_transforms.py
```

This generates a half-second stereo tone at 44,100 Hz, loads it, and converts it to
8,000 mono frames at 16,000 Hz. It checks that the 500 Hz pitch and half-second
duration remain correct and that the result reaches −20 dBFS RMS. The source is
temporary and removed afterward. No microphone or real recording is used.
Run `./scripts/verify_project.ps1` for the full project checks.

## In plain language

Mono conversion combines the audio channels into one. Resampling puts the audio
onto a common timing grid without speeding it up or slowing it down. Normalization
adjusts the volume toward a common level, with limits on amplification and peaks.
Together they give later analysis stages consistently formatted audio to work with.
