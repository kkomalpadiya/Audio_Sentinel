# A2.1: Log-Mel feature contract

A2.1 defines validated settings and metadata for one prepared window's numeric
features. [B2.1 computes them in memory](log-mel-generation.md),
[A2.2 stores and verifies them](feature-storage.md),
[B2.2 expands numerical property tests](feature-property-tests.md), and A2.3 will
connect extraction to the preparation service.

## Input and default recipe

Use a saved Phase 1 window WAV, decoded to float32, at the recipe's sample rate.
Version 1 accepts mono only. Reject stereo or mismatched rates at extraction;
do not silently resample or mix again. Preparation-only runs keep their existing
stereo and sample-rate options. These defaults are a project feature recipe, not
a claim of compatibility with a particular pretrained model. A3.1 must check the
selected model's exact frontend, including any internal feature extraction.

| Setting | Default |
| --- | --- |
| Sample rate | 16,000 Hz |
| FFT length | 512 samples |
| Analysis window | 400 samples (25 ms), periodic Hann, centered inside the FFT frame |
| Hop | 160 samples (10 ms) |
| Center input frames | False |
| Mel filters | 64, Slaney scale and Slaney area normalization |
| Frequency range | 0 to 8,000 Hz |
| Spectrum | Unnormalized FFT magnitude squared (power 2) |
| Log conversion | Power dB, fixed reference 1.0, minimum power 1e-10 |
| Dynamic range | 80 dB below the maximum within this window's feature matrix |
| Output | C-contiguous float32, axes `(mel, time)`, `.npy` without pickle |
| Maximum array payload | 256 MiB per window |

Configuration: `configs/log-mel.example.json`. `LogMelSettings` rejects unknown
fields, invalid frequency ranges, inconsistent FFT/window/hop lengths, and
non-finite numeric settings. Generation must additionally reject empty Mel
filters for unusual custom frequency/FFT combinations, invalid input samples,
and non-finite output; configuration validation alone cannot verify an array.
The payload limit does not cover temporary FFT workspace; B2.1 must bound that
workspace before allocation too.

## Time grid and padding

Let `N` include the preparation window's existing tail padding. If `N < n_fft`,
append exactly `n_fft - N` zeros on the right. Otherwise add no analysis padding.
Keep complete FFT frames and drop a trailing incomplete FFT frame:

`T = 1 + floor((max(N, n_fft) - n_fft) / hop_length)`

The shape is `(n_mels, T)`. Default 1, 5, and 10 second windows yield `(64, 97)`,
`(64, 497)`, and `(64, 997)`. Do not confuse these analysis frames with Phase 1's
1/5/10 second audio windows. `LogMelSettings.expected_shape(N)` validates the
payload budget without allocating an array.

Frame `t` has FFT start sample `window.start_sample + t * hop_length` relative
to the full prepared clip. Its nominal FFT center is that start plus `n_fft / 2`.
The periodic Hann window is zero-padded symmetrically inside the FFT frame
(for an odd difference, put the extra zero on the right). Timing is derived in
samples and divided by sample rate only for display. Preparation padding and
analysis padding remain separate metadata fields. Frames can contain padded
samples; clip event timestamps must stay inside `window.end_sample`. A dropped
FFT tail has no separate frame.

## Scaling and numeric values

For Mel power `S`, compute `D = 10*log10(max(S, amin)) -
10*log10(reference_power)`, then floor at `max(D) - top_db` over that window.
There is no peak-reference normalization, z-score normalization, image scaling,
or PNG encoding. Values may be positive; the contract does not assert a fixed
`[-80, 0]` range. With defaults, complete silence is finite at -100 dB.

The intended backend semantics are documented by librosa:
[Mel spectrogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html),
[Mel filter bank](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html), and
[power-to-dB](https://librosa.org/doc/0.11.0/generated/librosa.power_to_db.html).
Record the actual installed backend version when generating artifacts.
Repeatability is expected with the same input bytes, recipe, and software
environment; bitwise equality across library/platform versions is not promised.

## Metadata and source linkage

`LogMelFeatureMetadata` describes one `.npy` artifact. It stores the full recipe,
shape, dtype, axes, additional analysis padding, extractor and implementation
versions, timezone-aware creation time, feature path and SHA-256. `LogMelSource`
stores the source clip ID, original raw-file hash, full `PreparedWindowRecord`,
window WAV hash, preparation manifest path/hash, sample rate, and channel count.

Feature paths are relative to `data/processed`; manifest and window paths are
relative to `data/interim`. All are portable relative POSIX paths. Hashes refer
to exact file bytes, not decoded samples or a reserialized JSON document.
The preparation manifest remains authoritative for dataset, annotations,
consent, and preprocessing history. `source.validate_against(manifest)` checks
clip/hash/rate/channels and exact window membership. Metadata parsing does not
grant permission or prove files exist. The future runtime must resolve paths
with containment checks, verify hashes and actual shapes/dtypes, and recheck
current consent before processing or consuming saved features.

JSON Schemas capture field-level constraints; Pydantic validators additionally
enforce relationships between fields. Consumers must run those relationship
checks too. The example metadata has placeholder hashes and a demonstrative
backend version; it does not represent a generated feature file.

```python
from pathlib import Path
from audio_sentinel.config import load_settings
from audio_sentinel.features import LogMelFeatureMetadata

settings = load_settings(log_mel_config_path=Path("configs/log-mel.example.json"))
print(settings.log_mel.expected_shape(16_000))  # (64, 97)
metadata = LogMelFeatureMetadata.model_validate_json(
    Path("docs/examples/log-mel-feature.json").read_text()
)
```

## In plain language

Phase 1 prepared the audio. This task defines the format of the frequency-over-time
numbers we will derive from each audio window, plus a record tying those numbers
back to the original recording. B2.1 will perform that conversion. No downloads,
recordings, model installation, or manual data processing are needed for A2.1.
