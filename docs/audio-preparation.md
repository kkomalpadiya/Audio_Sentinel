# A1.1: Audio preparation settings and manifest

A1.1 defines the rules that later preparation modules must follow. The settings,
manifest validation, JSON schemas, and examples are implemented. Audio loading,
conversion, noise reduction, segmentation, and writing audio files remain the
following tasks. No dataset download or microphone access is needed for A1.1.

## Settings

`AudioSentinelSettings.audio` remains the shared configuration passed to the
`AudioPreprocessor` interface. `AudioSettings` now defines:

| Setting | Default | Meaning |
| --- | --- | --- |
| Input formats | WAV, FLAC | Check the decoded container format, not only the extension. |
| Input limits | 256 MiB, 600 seconds, 8 channels | Reject empty, oversized, unsupported, corrupt, or non-finite audio in B1.1. Source rate must be 8,000–192,000 Hz. |
| Sample rate | 16,000 Hz | Keep the Phase 0 default; adjustable from 8,000 to 192,000 Hz. |
| Mono conversion | Enabled | Average channels. If disabled, preserve channel count. |
| Volume normalization | Enabled, RMS target −20 dBFS | Adjust average signal level with one gain for the entire clip. |
| Gain limits | +20 dB maximum, −1 dBFS peak ceiling | Limit amplification and prevent peaks exceeding the ceiling. |
| Silence floor | −60 dBFS | Leave clips at or below this RMS level unchanged. |
| Noise reduction | Disabled; strength 0.5 when enabled | B1.2 will implement stationary spectral gating using a noise estimate from the clip. |
| Output | WAV, signed 16-bit PCM | Full clip and window files share this format. |
| Window durations | 1, 5, 10 seconds | Three views of the same prepared clip. |
| Overlap | 0.5 | Advance by half a window. Zero disables overlap. |
| Tail handling | `pad` | Add zeros at the end of the final incomplete window. `drop` omits incomplete windows. |

These are initial project choices, not measured optimal values. Later evaluation
can tune them. RMS normalization is not perceptual LUFS normalization.

Use defaults with `load_settings()`. To load the checked-in example explicitly:

```python
from pathlib import Path
from audio_sentinel.config import load_settings

settings = load_settings(audio_config_path=Path("configs/preprocessing.example.json"))
```

Relative config paths resolve from the project root. Missing, malformed, or invalid
files raise errors. Unknown keys are rejected to catch spelling mistakes.
The example is not loaded automatically. Settings are frozen after validation.

## Processing contract for the next tasks

1. B1.1 loads a local, authorized file as finite floating-point samples, scales
   integer PCM to full-scale units, and checks input limits before large allocations.
2. A1.2 averages channels if enabled and resamples. The output frame count is
   `round(source_frames * target_rate / source_rate)`; trim/pad the resampler output
   to that count if needed. Python rounding uses ties to even.
3. B1.2 optionally applies noise reduction at the target rate, preserving length
   and channel count. Disabled means bypass. Algorithm details belong to B1.2.
4. A1.2 applies normalization after noise reduction when the steps are integrated.
   Compute RMS over all samples and channels before window padding. For RMS above
   the silence floor, apply a single gain in dB:
   `min(target_rms_dbfs - rms_dbfs, max_gain_db, peak_ceiling_dbfs - peak_dbfs)`.
   A negative gain attenuates. Silent clips stay unchanged; disabled normalization
   means bypass. Clip to the PCM representable range only at encoding time.
5. B1.3 segments the prepared full clip using the sample grid below.
6. A1.3 persists the full clip, windows, and manifest; A1.4 connects these steps.

All annotation times remain relative to the original full clip. Preparation does
not trim silence or change playback speed. Preserve consent and clip ID through
the existing `InputAudio` → `PreprocessedAudio` interface.

## Window grid

At the target rate, window length `L = round(seconds * sample_rate)` and hop
`H = round(L * (1 - overlap))`. Both must be at least one frame. A frame means one
sample per channel, so stereo does not double the timeline count.

Starts are `0, H, 2H, ...`. With `pad`, stop at the first window whose requested end
reaches or passes the clip end. A clip shorter than a window produces one padded
window. With `drop`, keep only complete windows. Do not emit extra overlapping tails
after an exact fit. Never generate an all-padding window.

For a 1.6-second clip at 16 kHz with 1-second windows and 50% overlap:

| Start frame | End frame (exclusive) | Added zero frames |
| --- | --- | --- |
| 0 | 16,000 | 0 |
| 8,000 | 24,000 | 0 |
| 16,000 | 25,600 | 6,400 |

The original clip has 25,600 frames. Padding belongs to each window only and must
not extend the clip duration or annotations. Sort records by duration, then start.

## Manifest

`PreparedAudioManifest` in `src/audio_sentinel/preparation.py` wraps the existing
`PreparedClipRecord` without changing its v1 schema or consumers. It contains:

- `schema_version` and `preprocessing_version`: both `1.0` for this contract.
- `source`: relative raw-file path, SHA-256 of original file bytes, decoded format,
  sample rate, channel count, frame count, and file size.
- `settings`: the complete effective `AudioSettings` snapshot, including defaults.
- `clip`: the existing clip ID, dataset, relative prepared path, target rate,
  duration, consent, and annotations.
- `channels` and `num_frames`: properties of the full prepared clip.
- `windows`: unique IDs and paths, duration group, start/end frames, and padding.

Source paths are relative to `data/raw/`. Prepared paths are relative to
`data/interim/`, including the full clip and windows. Use forward slashes on every
OS. Absolute paths, drive names, backslashes, empty segments, and traversal are
rejected. Actual filesystem containment, symlinks, file existence, permission
expiry at processing time, and hashing are responsibilities of the loader and
persistence implementations. No transcripts, raw samples, or personal details
belong in manifests. The example uses fictional identifiers and a placeholder hash.

The validator checks source limits, rate and channel consistency, resampled frame
count, duration, complete window inventory, sample offsets, padding, unique paths,
and existing consent/annotation rules. JSON Schema captures the field shapes and
ranges; Python validation also checks relationships between fields. Consumers must
perform those relationship checks rather than relying on JSON Schema alone.

Examples: `configs/preprocessing.example.json` and
`docs/examples/prepared-audio-manifest.json`. Schema exports are in `docs/schemas/v1/`.
To regenerate only A1.1 schemas from the project root:

```powershell
$env:PYTHONPATH = "src"
python -c "from pathlib import Path; from audio_sentinel.preparation import write_preparation_schemas; write_preparation_schemas(Path('docs/schemas/v1'))"
```

## In plain language

The settings are a recipe: they tell future code how to prepare a recording. The
manifest is the record kept with the prepared audio: it says which recording was
used, which recipe was applied, and where each small audio window belongs. Validation
checks that this record makes sense before later modules rely on it. A1.1 creates
the recipe and record format; the following tasks build the code that processes audio.
